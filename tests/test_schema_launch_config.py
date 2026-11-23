"""Public launch policy and deferred input binding, without model or GPU calls."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aka.contracts.startup import Invocation
from aka.contracts.workflow import WorkflowSettings
from aka.contracts.workflow_launch import decode_launch, encode_launch
from aka.task.candidate_workspace.provider import GitCandidateWorkspace
from aka.contracts.workspace import RevisionContent
from aka.bootstrap.profile import resolve_profile
from aka.bootstrap.workflow_cli import profile_path
from aka.workflow.legacy.requests import create_launch
from aka.workflow.schema.launch import SchemaConfig, prepare_selection
from aka.workflow.schema.language import CheckPolicy, collect_design
from aka.workflow.schema.materials import read_design, snapshot_inputs
from aka.workflow.schema.workflow import SuppliedStart

FIXTURE = Path(__file__).parent / 'fixtures/schema_language/minimal'


def selected(tmp_path):
    selection = resolve_profile(profile_path('light-sandbox'), variables={'repo_root': str(tmp_path)})
    launch = create_launch(tmp_path / 'problem', workspace_base=tmp_path / 'runs',
        settings=WorkflowSettings(platform='cpu', framework='triton'), provided_settings=('platform', 'framework'))
    return selection, Invocation(encode_launch(launch))


def test_continuous_sets_native_policy_and_preserves_selected_resources(tmp_path):
    selection, invocation = selected(tmp_path)
    result, request = prepare_selection(selection, invocation, {'continuous': True})
    settings = decode_launch(request.argv).request.settings
    assert settings.max_iters == 0 and settings.framework_baseline == 'never'
    assert settings.convert_after == 0 and settings.setup_timeout == 3600
    assert result.resources == selection.resources
    assert result.submodules == selection.submodules
    for name in ('agent', 'problem', 'source', 'execution', 'resources'):
        assert result.composition.row(name) == selection.composition.row(name)
    assert result.composition.row('workflow').name == 'aka.workflow.schema.plugin'


def test_no_external_plugins_omit_only_unused_stock_wiki_manifest(tmp_path):
    selection, invocation = selected(tmp_path)
    resources = selection.composition.row('resources')
    configured = replace(resources, config={'plugins': [], 'bundles': ['measurement-session', 'research-records']})
    selection = replace(selection, composition=replace(selection.composition, rows=tuple(
        configured if row.id == 'resources' else row for row in selection.composition.rows)),
        resources=((tmp_path, ('plugins/gpu-wiki', 'gpu-wiki', 'retained-input')),),
        submodules={str(tmp_path): {'gpu-wiki': 'a' * 40, 'retained-input': 'b' * 40,
                                    'separate-dependency': 'c' * 40}})
    result, _ = prepare_selection(selection, invocation, {'continuous': True})
    assert result.composition.row('resources') == configured
    assert result.resources == ((tmp_path, ('retained-input',)),)
    assert result.submodules == {str(tmp_path): {'retained-input': 'b' * 40,
                                                'separate-dependency': 'c' * 40}}


def test_independent_submodule_pin_survives_unchanged_paths_and_capture(tmp_path, monkeypatch):
    from aka.bootstrap.continuation import capture, restore
    monkeypatch.setattr('aka.agent.execution.bubblewrap.executable', lambda *args, **kwargs: '/bin/true')
    monkeypatch.setattr('aka.agent.execution.proc_view.validate_runtime', lambda *args, **kwargs: None)
    selection, invocation = selected(tmp_path)
    root = tmp_path / 'other'
    (root / 'data').mkdir(parents=True)
    (root / 'data/input.txt').write_text('public CPU input\n')
    selection = replace(selection, resources=((root, ('data',)),),
        submodules={str(root): {'retained-dependency': 'a' * 40}},
        environment={key: None for key in selection.environment})
    assert selection.composition.row('resources').config['plugins']
    _, before = capture(selection)
    assert capture(restore(before))[1] == before
    transformed, _ = prepare_selection(selection, invocation, {'continuous': True})
    assert transformed.resources == selection.resources
    assert transformed.submodules == selection.submodules
    _, after = capture(transformed)
    assert capture(restore(after))[1] == after


@pytest.mark.parametrize('relative', ['', 'gpu-wiki', 'gpu-wiki/imports'])
def test_selected_python_paths_keep_their_resource_manifest_for_restore(tmp_path, monkeypatch, relative):
    from aka.bootstrap.continuation import capture, restore
    (tmp_path / 'plugins/gpu-wiki').mkdir(parents=True)
    (tmp_path / 'gpu-wiki').mkdir()
    imports = tmp_path / relative
    imports.mkdir(parents=True, exist_ok=True)
    (imports / 'selected_cpu_module.py').write_text('VALUE = 1\n')
    profile = json.loads(profile_path('light-sandbox').read_text())
    profile['python_paths'] = [str(imports)]
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(profile))
    (tmp_path / 'compositions').symlink_to(profile_path('light-sandbox').parent / 'compositions')
    selection = resolve_profile(path, patches=[{'id': 'resources', 'config': {
        'plugins': [], 'bundles': ['measurement-session', 'research-records']}}],
        variables={'repo_root': str(tmp_path)})
    selection = replace(selection, environment={key: None for key in selection.environment})
    monkeypatch.setattr('aka.agent.execution.bubblewrap.executable', lambda *args, **kwargs: '/bin/true')
    monkeypatch.setattr('aka.agent.execution.proc_view.validate_runtime', lambda *args, **kwargs: None)
    _, before = capture(selection)
    assert capture(restore(before))[1] == before
    _, invocation = selected(tmp_path)
    transformed, _ = prepare_selection(selection, invocation, {'continuous': True})
    _, after = capture(transformed)
    assert capture(restore(after))[1] == after
    assert transformed.python_paths == selection.python_paths


def test_continuous_preserves_explicit_finite_native_budget(tmp_path):
    selection, invocation = selected(tmp_path)
    launch = decode_launch(invocation.argv)
    launch = replace(launch, request=replace(launch.request,
        settings=replace(launch.request.settings, max_iters=7)), provided_settings=(*launch.provided_settings, 'max_iters'))
    _, request = prepare_selection(selection, Invocation(encode_launch(launch)),
                                   {'continuous': True, 'limits': {'max_windows': 3}})
    assert decode_launch(request.argv).request.settings.max_iters == 7


@pytest.mark.parametrize('old', [{'mode': 'strict'}, {'selection': 'old.json'}, {'format': 'kernelschema.v1'}])
def test_retired_configuration_is_rejected_without_fallback(old):
    with pytest.raises(ValueError, match='[Rr]etired'):
        SchemaConfig.load(old)


def test_config_file_requires_current_format(tmp_path):
    path = tmp_path / 'old.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='Retired'):
        SchemaConfig.load(path)


def test_relative_pair_inputs_resolve_against_configuration(tmp_path):
    evidence = tmp_path / 'evidence.json'
    evidence.write_text('{"kind":"declared_pair"}')
    path = tmp_path / 'launch.json'
    path.write_text(json.dumps({'format': 'aka.schema.launch.v1', 'initial': {
        'schema': str(FIXTURE.resolve()), 'source_files': {'kernel.py': 'a' * 64},
        'hardware': 'L20D', 'evidence': ['evidence.json']}}))
    config = SchemaConfig.load(path)
    assert config.initial['evidence'] == [str(evidence)]


def test_pair_requires_native_source_and_does_not_infer_reference(tmp_path):
    selection, invocation = selected(tmp_path)
    evidence = tmp_path / 'evidence.json'
    evidence.write_text('{}')
    config = {'initial': {'schema': str(FIXTURE.resolve()), 'source_files': {'kernel.py': 'a' * 64},
                         'hardware': 'L20D', 'evidence': [str(evidence)]}}
    with pytest.raises(ValueError, match='explicitly selected native kernel'):
        prepare_selection(selection, invocation, config)


def supplied(tmp_path):
    raw = b'def kernel(): return 1\n'
    initial = SuppliedStart(collect_design(FIXTURE), {'kernel.py': hashlib.sha256(raw).hexdigest()},
        'L20D', {'000-evidence.json': b'{"status":"declared_pair"}\n'}, CheckPolicy(skip_hardware=True))
    campaign = SimpleNamespace(candidate_workspace=GitCandidateWorkspace(), sandbox_hardware='L20D')
    snapshot = SimpleNamespace(identity='prepared-native-task')
    materials = snapshot_inputs(tmp_path / 'materials', ())
    return initial, dict(campaign=campaign, snapshot=snapshot, source=RevisionContent({'kernel.py': raw}),
                         directory=tmp_path / 'controller', materials_snapshot=materials)


def test_supplied_pair_binds_prepared_native_task_and_retains_provenance(tmp_path):
    initial, args = supplied(tmp_path)
    start = initial.bind(**args)
    assert start.prepared_id == 'prepared-native-task'
    assert read_design(start.design).manifest['mode'] == 'import'
    assert read_design(start.design).manifest['prepared_id'] == start.prepared_id
    assert Path(start.evidence_refs[0]).read_bytes() == initial.evidence['000-evidence.json']
    assert initial.bind(**args) == start


def test_supplied_pair_rejects_prepared_source_or_hardware_mismatch(tmp_path):
    initial, args = supplied(tmp_path)
    with pytest.raises(ValueError, match='full prepared native source'):
        initial.bind(**{**args, 'source': RevisionContent({'kernel.py': b'wrong'})})
    args['campaign'].sandbox_hardware = 'H100'
    with pytest.raises(ValueError, match='target hardware'):
        initial.bind(**args)


def test_initial_publication_interruption_retains_partial_and_rebuilds_same_bytes(tmp_path):
    initial, args = supplied(tmp_path)
    interrupted = args['directory'] / 'initial-design'
    interrupted.mkdir(parents=True)
    (interrupted / 'intent.json').write_text('{}')
    start = initial.bind(**args)
    assert start.design.directory.name == 'initial-design-1'
    assert (interrupted / 'intent.json').read_text() == '{}'
    assert read_design(start.design).checked.files == initial.files


def test_cli_schema_path_completes_pair_config_and_keeps_relative_evidence(tmp_path):
    evidence = tmp_path / 'pair.json'
    evidence.write_text('{}')
    path = tmp_path / 'launch.json'
    path.write_text(json.dumps({'format': 'aka.schema.launch.v1', 'initial': {
        'source_files': {'kernel.py': 'a' * 64}, 'hardware': 'L20D', 'evidence': ['pair.json']}}))
    config = SchemaConfig.load(path, initial_schema=FIXTURE)
    assert config.initial['schema'] == str(FIXTURE.resolve())
    assert config.initial['evidence'] == [str(evidence)]
