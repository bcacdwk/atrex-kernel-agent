"""Real NativeSession/fake-process file output tests for serial AKA extraction."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from aka.agent.provider import Agent, AgentOptions
from aka.contracts.agent import AgentCapabilities
from aka.workflow.schema import materials
from aka.workflow.schema.materials import DesignRef
from aka.workflow.schema.extract_sources import (SourceDescriptor, SourceLocator, SourceQuote,
    GPUClaim, OutputComparator, SourceError, inspect_sources)
from aka.workflow.schema.extract_stages import (Extractor, ExtractRequest, ExtractConfig,
    approval_projection, check_scope, check_stage_snapshot, validate_plan, defining_stage)
from aka.workflow.schema.planner import Planner, PlannerConfig
from aka.workflow.schema.run_state import RecoveryRequired, RunLimits
from aka.workflow.schema.workflow import SchemaWorkflow, SotaStart

FIXTURE = Path(__file__).parent / 'fixtures/schema_language/minimal'
TARGET = {'model': 'NVIDIA L20D', 'kind': 'declaration', 'origin': 'environment', 'evidence': 'Environment selects NVIDIA L20D'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class CpuAgent(Agent):
    @property
    def capabilities(self):
        return AgentCapabilities(filesystem_isolation=True, tool_runtime=True)


def quote(inputs, name, start=1, end=None):
    raw = inputs[name]
    lines = raw.decode().splitlines()
    end = start if end is None else end
    return {'path': name, 'line_start': start, 'line_end': end,
            'quote': '\n'.join(lines[start - 1:end]), 'sha256': sha(raw)}


def references(inputs, role):
    manifest = json.loads(inputs['sources/manifest.json'])
    return ['aka-source@' + row['sha256'] + ':' + row['path'] +
            ('::' + row['symbol'] if row['symbol'] else '#L%d-L%d' % (row['line_start'], row['line_end']))
            for row in manifest['roles'][role]]


def evidence(inputs, key=None):
    role = {'algorithm': 'reference', 'io': 'io', 'domain': 'scope', 'comparators': 'acceptance',
            'execution': 'scope', 'instances': 'implementation', 'source_fidelity': 'implementation'}.get(key, 'scope')
    if key and key.startswith('comparator:'):
        role = 'acceptance'
    row = json.loads(inputs['sources/manifest.json'])['roles'][role][0]
    return [quote(inputs, 'sources/' + row['path'], row['line_start'], min(row['line_end'], row['line_start'] + 2))]


class Scenario:
    def __init__(self):
        self.calls, self.admissions, self.hooks = [], [], {}
        self.multi = False
        self.ready = []
        self.optimize_files = None

    def process(self, command, *, cwd, timeout, env):
        assert command[0] == 'claude' and env['ANTHROPIC_MODEL'] == 'qwen3.8-max[1m]'
        assert timeout == 7
        workspace = Path(cwd)
        inputs = {p.relative_to(workspace / 'inputs').as_posix(): p.read_bytes()
                  for p in (workspace / 'inputs').rglob('*') if p.is_file()}
        if self.optimize_files is not None:
            choice = {'status': 'replace', 'rationale': 'Preserve explicit hardware scope and refine scalar computation',
                'hypothesis_key': 'scalar-refinement', 'changed_execution_decisions': ['Add an explicit zero identity'],
                'changes': [{'path': 'root.DESIGN.Math_reshape', 'before': 'y = x', 'after': 'y = x + 0', 'reason': 'Make identity explicit'}],
                'evidence': [{'source': 'task/public.md', 'start': 1, 'end': 1, 'claim': 'Scalar copy Task'}],
                'next_actions': ['Check exact scalar equality']}
            (workspace / 'output/decision.json').write_text(json.dumps(choice))
            for name, raw in self.optimize_files.items():
                (workspace / 'output/design' / name).write_bytes(raw)
            return json.dumps({'type': 'result', 'result': 'completed', 'usage': {'input_tokens': 11, 'output_tokens': 7}}), '', 0, False
        context = json.loads(inputs['task/stage.json'])
        binding = context['binding']
        stage, node, role = binding['stage'], binding['node'], binding['role']
        self.calls.append((stage, node, role, workspace))
        selection, plan = context['selection'], context['writing_plan']
        knowledge = [{'finding': 'Synthetic scalar mechanism', 'destination': 'test design', 'rationale': 'CPU evidence only'}] if stage == 'P6' else []
        if role == 'reviewer':
            files = {}
            receipt = {'binding': binding, 'decision': 'approve', 'checks': {
                key: {'status': 'pass', 'rationale': 'Synthetic exact evidence for ' + key,
                      'evidence': evidence(inputs, key)} for key in context['required_checks']},
                'revision': None, 'knowledge': knowledge}
        else:
            files = {name.removeprefix('current-design/'): content for name, content in inputs.items()
                     if name.startswith('current-design/')}
            root = yaml.safe_load(files['root.yaml'])
            fixture = yaml.safe_load((FIXTURE / 'root.yaml').read_text())
            if stage == 'P1':
                root['META_DATA'].update(Name=fixture['META_DATA']['Name'], Describe=fixture['META_DATA']['Describe'], Version=2)
                root['CONTRACT']['Algorithm_requirement'] = fixture['CONTRACT']['Algorithm_requirement']
                root['CONTRACT']['Algorithm_requirement']['Golden_reference'] = {
                    'Source': references(inputs, 'reference')[0],
                    'Tolerance': {'y': 'atol=0, rtol=0, relative_to=reference, mode=elementwise; abs(candidate-reference) <= 0 + 0 * abs(reference)'}}
                root['CONTRACT']['Hardware_specification']['Execution'] = fixture['CONTRACT']['Hardware_specification']['Execution']
                files['root.yaml'] = yaml.safe_dump(root, sort_keys=False).encode()
            elif stage == 'P2':
                root['META_DATA'].update(Source=references(inputs, 'implementation')[0], Dispatch={'id': 'root'}, Version=3)
                selection = [{'id': 'root', 'instance': references(inputs, 'implementation')[0]}]
                files['dispatch.md'] = (FIXTURE / 'dispatch.md').read_bytes()
                if self.multi:
                    selection.append({'id': 'other', 'instance': references(inputs, 'implementation')[1]})
                    files['dispatch.md'] += '| 仅计划 other.yaml | Another scalar instance for the alternate case | Same Contract |\n'.encode()
                files['root.yaml'] = yaml.safe_dump(root, sort_keys=False).encode()
            elif stage == 'P3':
                root['DESIGN'] = fixture['DESIGN']
                root['META_DATA']['Version'] += 1
                root['META_DATA']['Dispatch']['when'] = 'scalar_count == 1' if self.multi else 'true'
                files['root.yaml'] = yaml.safe_dump(root, sort_keys=False).encode()
            elif stage == 'P4':
                plan = [{'id': 'other', 'wait_for': ['root']}] if self.multi else []
            elif stage == 'P5':
                other = {'META_DATA': {**root['META_DATA'], 'Version': 1, 'Source': references(inputs, 'implementation')[1],
                    'Dispatch': {'id': node, 'when': 'scalar_count > 1'}},
                    'DESIGN': {key: {'from': 'root'} for key in fixture['DESIGN']}}
                files[node + '.yaml'] = yaml.safe_dump(other, sort_keys=False).encode()
            elif stage == 'P6':
                files = {}
            receipt = {'binding': binding, 'decision': 'complete', 'evidence': evidence(inputs),
                       'selection': selection, 'writing_plan': plan, 'unresolved': [], 'revision': None,
                       'knowledge': knowledge}
        hook = self.hooks.get((stage, role))
        if hook:
            hook(inputs, context, files, receipt)
        (workspace / 'output/decision.json').write_text(json.dumps(receipt))
        for name, raw in files.items():
            path = workspace / 'output/design' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        return json.dumps({'type': 'result', 'result': 'completed', 'usage': {'input_tokens': 11, 'output_tokens': 7}}), '', 0, False


@pytest.fixture
def harness(tmp_path):
    original = tmp_path / 'original'
    original.mkdir()
    data = {'kernel.py': 'def kernel(x):\n    return x\n\ndef other(x):\n    return x\n',
            'reference.py': 'def reference(x):\n    return x\n',
            'scope.md': 'Synthetic CPU fixture, no real GPU measurements.\nDevice: NVIDIA L20D\nInput scalar x; output scalar y; one exact copy.\n',
            'acceptance.json': '{"y":{"atol":0,"rtol":0,"relative_to":"reference","mode":"elementwise"}}\n'}
    for name, value in data.items():
        (original / name).write_text(value)
    loc = lambda name, symbol=None: SourceLocator(original / name, symbol=symbol)
    raw_acceptance = data['acceptance.json'].encode()
    comparator = OutputComparator('y', 'y', {'atol': 0, 'rtol': 0, 'relative_to': 'reference', 'mode': 'elementwise'},
        (SourceQuote('acceptance.json', 1, 1, data['acceptance.json'], sha(raw_acceptance)),), '/y')
    descriptor = SourceDescriptor({'implementation': (loc('kernel.py', 'kernel'), loc('kernel.py', 'other')),
        'host': (loc('kernel.py'),), 'io': (loc('scope.md'),), 'input': (loc('scope.md'),),
        'scope': (loc('scope.md'),), 'reference': (loc('reference.py', 'reference'),),
        'acceptance': (loc('acceptance.json'),)}, ('y',), (comparator,))
    source, _ = inspect_sources(descriptor)
    descriptor = replace(descriptor, source_gpu=GPUClaim('NVIDIA L20D', 'declaration', source['source_digest'],
        source['scope_digest'], SourceQuote('scope.md', 2, 2, 'Device: NVIDIA L20D', sha(data['scope.md'].encode()))))
    scenario = Scenario()
    snapshot = materials.snapshot_inputs(tmp_path / 'materials', materials.select_materials(wiki_ids=[]))
    agent = CpuAgent(AgentOptions(state_directory=tmp_path / 'agent-state'), process_runner=scenario.process,
        environment={'HOME': str(tmp_path), 'PATH': '/usr/bin:/bin', 'ANTHROPIC_MODEL': 'qwen3.8-max[1m]'})
    usage = []
    planner = Planner(agent, snapshot, config=PlannerConfig(timeout_seconds=7),
                      observe_result=lambda *args: usage.append(args))
    extractor = Extractor(planner)
    request = ExtractRequest('prepared-cpu', 'extract-1', 1, tmp_path / 'extract', 'Copy scalar x to y.\n',
                             'NVIDIA L20D', descriptor, TARGET)
    def admit(**kwargs):
        scenario.admissions.append(kwargs)
        return len(scenario.admissions)
    with agent.invocation_scope(admit=admit, observe=lambda *args, **kwargs: None):
        yield SimpleNamespace(extractor=extractor, planner=planner, request=request, scenario=scenario, agent=agent, usage=usage)
    agent.close()


def state(h):
    return json.loads((h.request.directory / 'state.json').read_text())


def test_single_root_all_stages_fresh_native_sessions_raw_and_projection_replay(harness):
    h = harness
    result = h.extractor.extract(h.request)
    assert result.status == 'replace', result.diagnostics
    assert result.calls == 10
    assert [stage for stage, _, role, _ in h.scenario.calls if role == 'worker'] == ['P1', 'P2', 'P3', 'P4', 'P6']
    assert len({usage[1].session_id for usage in h.usage}) == 10
    artifact = materials.read_design(result.design_ref)
    assert artifact.manifest['mode'] == 'extract' and artifact.manifest['check_policy'] == {'skip_hardware': True, 'style': 'strict'}
    assert len(artifact.manifest['approval_refs']) == 5
    assert h.extractor.extract(h.request) == result and len(h.scenario.calls) == 10
    for stage, _, role, workspace in h.scenario.calls:
        access = next(i['access'] for i in h.scenario.admissions if i['workspace'] == workspace)
        assert workspace / 'inputs' in access.read_only_paths
        if role == 'reviewer' or stage == 'P6':
            assert workspace / 'output/design' in access.read_only_paths
        assert (workspace.parent / 'native/receipt.json').is_file()
        assert (workspace.parent / 'raw/decision.json').is_file()


def test_multi_node_p3_accepts_unwritten_plan_and_p5_registered_only_after_review(harness):
    h = harness
    h.scenario.multi = True
    result = h.extractor.extract(h.request)
    assert result.status == 'replace', result.diagnostics
    assert result.calls == 12
    current = state(h)
    p3 = json.loads(Path(current['approvals'][2]['path']).read_text())
    assert set(p3['checked']['files']) == {'root.yaml', 'dispatch.md'}
    assert '仅计划 other.yaml' in (Path(p3['checked']['directory']) / 'dispatch.md').read_text()
    artifact = materials.read_design(result.design_ref)
    assert set(artifact.checked.files) == {'root.yaml', 'other.yaml', 'dispatch.md'}
    assert '[other](other.yaml)' in artifact.checked.files['dispatch.md'].decode()
    assert '仅计划' not in artifact.checked.files['dispatch.md'].decode()
    assert artifact.checked.resolved.origins['other']['Math_reshape'].provider == 'root'
    p1 = json.loads(Path(current['approvals'][0]['path']).read_text())
    assert p1['projection'] == approval_projection('P1', artifact.checked.files)
    assert p1['checked']['files'] != {name: sha(raw) for name, raw in artifact.checked.files.items()}


@pytest.mark.parametrize('stage,kind', [('P1', 'design'), ('P2', 'contract'), ('P3', 'add'), ('P4', 'delete'), ('P5', 'contract'), ('P6', 'edit')])
def test_worker_stage_scope_rejects_real_edits_additions_deletions(harness, stage, kind):
    h = harness
    h.scenario.multi = stage == 'P5'
    def bad(inputs, context, files, receipt):
        if kind == 'delete':
            files.pop('root.yaml')
        elif kind == 'add':
            files['extra.yaml'] = files['root.yaml']
        elif kind == 'edit':
            files['root.yaml'] = inputs['current-design/root.yaml']
        else:
            name = 'other.yaml' if stage == 'P5' else 'root.yaml'
            doc = yaml.safe_load(files[name])
            if kind == 'design':
                doc['DESIGN']['Math_reshape']['Thesis'] = 'Illegal early implementation'
            elif stage == 'P5':
                doc['CONTRACT'] = {}
            else:
                doc['CONTRACT']['Algorithm_requirement']['Compute'][0]['Math'] = 'y = x + 1'
            files[name] = yaml.safe_dump(doc, sort_keys=False).encode()
    h.scenario.hooks[(stage, 'worker')] = bad
    result = h.extractor.extract(h.request)
    assert result.status == 'failed'
    assert any(s == stage and role == 'worker' for s, _, role, _ in h.scenario.calls)
    assert not (h.request.directory / 'published/manifest.json').exists()
    assert sum(s == stage and role == 'reviewer' for s, _, role, _ in h.scenario.calls) == 0


@pytest.mark.parametrize('field,value', [('nonce', 'stale'), ('operation_id', 'other'), ('candidate_digest', '0' * 64), ('role', 'worker')])
def test_independent_reviewer_rejects_stale_or_foreign_binding(harness, field, value):
    h = harness
    h.scenario.hooks[('P1', 'reviewer')] = lambda i, c, f, r: r['binding'].__setitem__(field, value)
    assert h.extractor.extract(h.request).status == 'failed'
    assert not state(h)['approvals']


def test_reviewer_cannot_write_and_missing_conclusion_is_not_approval(harness):
    h = harness
    def bad(i, c, files, receipt):
        files['root.yaml'] = i['candidate/root.yaml']
        receipt['checks'].pop('io')
    h.scenario.hooks[('P1', 'reviewer')] = bad
    assert h.extractor.extract(h.request).status == 'failed'


def test_exact_quote_drift_rejected(harness):
    h = harness
    def bad(i, c, f, r):
        r['checks']['algorithm']['evidence'][0]['quote'] = 'Invented source'
    h.scenario.hooks[('P1', 'reviewer')] = bad
    assert h.extractor.extract(h.request).status == 'failed'


def test_upstream_revision_revokes_successors_and_reexecutes_defining_stage(harness):
    h = harness
    used = []
    def revise_once(i, c, files, receipt):
        if not used:
            used.append(True)
            receipt.update(decision='revise', revision={'path': 'root.CONTRACT.Algorithm_requirement', 'reason': 'Recheck scalar obligation'})
            receipt['checks']['contract']['status'] = 'fail'
    h.scenario.hooks[('P3', 'reviewer')] = revise_once
    result = h.extractor.extract(h.request)
    assert result.status == 'replace', result.diagnostics
    assert [s for s, _, r, _ in h.scenario.calls if r == 'worker'] == ['P1', 'P2', 'P3', 'P1', 'P2', 'P3', 'P4', 'P6']
    assert [r['stage'] for r in state(h)['revoked'][0]['approvals']] == ['P1', 'P2']
    assert all(ref['identity'] not in {r['identity'] for r in state(h)['revoked'][0]['approvals']}
               for ref in state(h)['approvals'])


def test_p6_revision_returns_to_root_definition_and_cannot_local_patch(harness):
    h = harness
    used = []
    def revise(i, c, f, r):
        if not used:
            used.append(True)
            r.update(decision='revise', revision={'path': 'root.DESIGN.Math_reshape', 'reason': 'Check copy semantics again'})
            r['checks']['source_fidelity']['status'] = 'unknown'
    h.scenario.hooks[('P6', 'reviewer')] = revise
    result = h.extractor.extract(h.request)
    assert result.status == 'replace', result.diagnostics
    assert [r['stage'] for r in state(h)['revoked'][0]['approvals']] == ['P3', 'P4']
    assert len(state(h)['approvals']) == 5


def test_p6_must_cover_every_node_and_knowledge_destination(harness):
    h = harness
    h.scenario.multi = True
    h.scenario.hooks[('P6', 'reviewer')] = lambda i, c, f, r: r['checks'].pop('node:other')
    assert h.extractor.extract(h.request).status == 'failed'
    assert all(r['stage'] != 'P6' for r in state(h)['approvals'])


@pytest.mark.parametrize('mutation', ['raw', 'approval', 'reviewer', 'input', 'worker_raw', 'native_raw'])
def test_replay_rejects_changed_retained_evidence_or_inputs_without_new_calls(harness, mutation):
    h = harness
    result = h.extractor.extract(h.request)
    assert result.status == 'replace', result.diagnostics
    approval_ref = state(h)['approvals'][0]
    approval = json.loads(Path(approval_ref['path']).read_text())
    if mutation == 'raw':
        path = Path(approval['raw']['directory']) / 'root.yaml'
    elif mutation == 'approval':
        path = Path(approval_ref['path'])
    elif mutation == 'reviewer':
        path = Path(approval['reviewer_directory']) / 'attempts/1/raw/decision.json'
    elif mutation == 'worker_raw':
        path = Path(approval['worker_directory']) / 'attempts/1/raw/design/root.yaml'
    elif mutation == 'native_raw':
        path = Path(approval['worker_directory']) / 'attempts/1/native/stdout.txt'
    else:
        path = None
        h.request = replace(h.request, public_task='A different task')
    if path:
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError):
        h.extractor.extract(h.request)
    assert len(h.scenario.calls) == 10


@pytest.mark.parametrize('mode', ['mismatch', 'unknown', 'target_backfill'])
def test_source_gpu_preflight_has_zero_agent_calls(harness, mode):
    h = harness
    if mode == 'mismatch':
        h.request = replace(h.request, target_gpu={**TARGET, 'model': 'NVIDIA H100', 'evidence': 'Environment NVIDIA H100'})
    else:
        gpu = None if mode == 'unknown' else replace(h.request.descriptor.source_gpu, origin='target')
        h.request = replace(h.request, descriptor=replace(h.request.descriptor, source_gpu=gpu))
    with pytest.raises(SourceError):
        h.extractor.extract(h.request)
    assert h.scenario.calls == [] and h.scenario.admissions == []


def test_unknown_started_native_role_never_relaunches(harness, monkeypatch):
    h = harness
    original = h.planner.retained_role
    def interrupted(**kwargs):
        # Let the real planner write the intent, then fail before receipt.
        monkeypatch.setattr(h.agent, 'open_session', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('lost native owner')))
        return original(**kwargs)
    monkeypatch.setattr(h.planner, 'retained_role', interrupted)
    with pytest.raises(RuntimeError, match='lost native owner'):
        h.extractor.extract(h.request)
    monkeypatch.setattr(h.planner, 'retained_role', original)
    with pytest.raises(RecoveryRequired, match='original native owner'):
        h.extractor.extract(h.request)
    assert h.scenario.calls == []


def test_plan_coverage_cycles_and_definition_routing():
    selection = [{'id': 'root', 'instance': 'root-call'}, {'id': 'a', 'instance': 'a-call'}, {'id': 'b', 'instance': 'b-call'}]
    with pytest.raises(ValueError, match='cover'):
        validate_plan('P4', [{'id': 'a', 'wait_for': ['root']}], selection=selection)
    with pytest.raises(ValueError, match='cycle'):
        validate_plan('P4', [{'id': 'a', 'wait_for': ['b']}, {'id': 'b', 'wait_for': ['a']}], selection=selection)
    assert defining_stage('root.CONTRACT.Algorithm_requirement') == ('P1', 'root')
    assert defining_stage('a.DESIGN.Tile_pipeline') == ('P5', 'a')


def test_allowed_normalization_is_independently_reviewed_and_replayed(harness):
    h = harness
    def casing(i, c, files, r):
        files['root.yaml'] = files['root.yaml'].replace(b'Math_reshape:', b'MATH_RESHAPE:')
    h.scenario.hooks[('P3', 'worker')] = casing
    result = h.extractor.extract(h.request)
    assert result.status == 'replace', result.diagnostics
    ref = state(h)['approvals'][2]
    approval = json.loads(Path(ref['path']).read_text())
    assert approval['normalization_digest']
    assert b'MATH_RESHAPE:' in (Path(approval['raw']['directory']) / 'root.yaml').read_bytes()
    assert b'Math_reshape:' in (Path(approval['checked']['directory']) / 'root.yaml').read_bytes()
    reviewer = next(workspace for s, _, role, workspace in h.scenario.calls if s == 'P3' and role == 'reviewer')
    assert b'MATH_RESHAPE:' not in (reviewer / 'inputs/candidate/root.yaml').read_bytes()
    assert h.extractor.extract(h.request) == result
    path = Path(ref['path']).parent / 'normalization.json'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='normalization transformation'):
        h.extractor.extract(h.request)


def test_restoring_after_retained_worker_raw_never_duplicates_model(harness, monkeypatch):
    h = harness
    original = h.extractor._checked
    once = []
    def interrupted(*args, **kwargs):
        if not once:
            once.append(True)
            raise RuntimeError('host interrupted after worker raw')
        return original(*args, **kwargs)
    monkeypatch.setattr(h.extractor, '_checked', interrupted)
    with pytest.raises(RuntimeError, match='host interrupted'):
        h.extractor.extract(h.request)
    assert len(h.scenario.calls) == 1
    result = h.extractor.extract(h.request)
    assert result.status == 'replace', result.diagnostics
    assert len(h.scenario.calls) == 10
    assert h.usage[0][0] == h.usage[1][0]


def test_completed_extract_replays_after_original_sources_disappear(harness):
    h = harness
    result = h.extractor.extract(h.request)
    assert result.status == 'replace', result.diagnostics
    for path in (h.request.directory.parent / 'original').iterdir():
        path.unlink()
    assert h.extractor.extract(h.request) == result
    assert len(h.scenario.calls) == 10


def test_p6_mechanical_gate_rejects_missing_selected_node_without_call(harness):
    h = harness
    h.scenario.multi = True
    result = h.extractor.extract(h.request)
    assert result.status == 'replace', result.diagnostics
    files = materials.read_design(result.design_ref).checked.files.copy()
    files.pop('other.yaml')
    with pytest.raises(ValueError, match='inventory'):
        check_stage_snapshot('P6', files, selection=state(h)['selection'], plan=state(h)['plan'])


def test_actual_from_provider_must_appear_in_writing_plan(harness):
    h = harness
    h.scenario.multi = True
    result = h.extractor.extract(h.request)
    assert result.status == 'replace', result.diagnostics
    files = materials.read_design(result.design_ref).checked.files.copy()
    other = yaml.safe_load(files['other.yaml'])
    third = deepcopy(other)
    third['META_DATA']['Dispatch']['id'] = 'third'
    third['DESIGN'] = {'Math_reshape': {'from': 'other'}, 'Work_schedule': {'from': 'other'}, 'Tile_pipeline': {'from': 'other'}}
    files['third.yaml'] = yaml.safe_dump(third, sort_keys=False).encode()
    files['dispatch.md'] += b'| [third](third.yaml) | Third synthetic case | Provider other |\n'
    selection = state(h)['selection'] + [{'id': 'third', 'instance': 'synthetic-third-instance'}]
    plan = state(h)['plan'] + [{'id': 'third', 'wait_for': ['root']}]
    with pytest.raises(ValueError, match='actual from providers'):
        check_stage_snapshot('P6', files, selection=selection, plan=plan)


def test_controller_explicit_sota_source_enters_extract_without_design(harness, monkeypatch):
    from test_schema_workflow import native, controller, FakePlanner
    h = harness
    root = h.request.directory.parent / 'controller'
    root.mkdir()
    campaign = native(root)
    planner = FakePlanner(root, campaign.agent)
    calls = []
    def extracting(request, *, planner, config=None):
        calls.append(request)
        ref = materials.publish_design(request.directory / 'published', FIXTURE, version=request.version,
            prepared_id=request.prepared_id, operation_id=request.operation_id, materials_snapshot=planner.materials_snapshot,
            mode='extract')
        from aka.workflow.schema.planner import PlannerResult
        return PlannerResult('replace', ref, request.operation_id, request.directory,
            hypothesis_key='explicit-sota', next_actions=('Implement retained source design',))
    monkeypatch.setattr('aka.workflow.schema.workflow.extract', extracting)
    driver, _ = controller(root, campaign, planner=planner, limits=RunLimits(max_episodes=1),
        sota=SotaStart(h.request.descriptor, TARGET, 'L20D'))
    assert driver.run(campaign) == 'budget: episodes'
    assert len(calls) == 1 and isinstance(calls[0], ExtractRequest)
    assert planner.calls == []
    assert driver.state.data['counts']['settled_episodes'] == 1


def test_controller_pair_and_source_only_inputs_cannot_silently_replace_each_other(harness):
    from aka.workflow.schema.workflow import ExistingStart
    from aka.execution.legacy.executor import LegacyExecutor
    h = harness
    initial = ExistingStart(DesignRef(Path('/unread'), 'a' * 64), 'source', 'prepared-cpu', 'L20D', ('pair-evidence',))
    with pytest.raises(ValueError, match='mutually exclusive'):
        SchemaWorkflow(h.request.directory.parent / 'controller', planner=h.planner,
            executor=LegacyExecutor(), workspace=h.request.directory.parent / 'workspace', limits=RunLimits(max_episodes=1),
            initial=initial, sota=SotaStart(h.request.descriptor, TARGET, 'L20D'))
    assert not h.scenario.calls


@pytest.mark.parametrize('bad', ['source', 'hardware', 'target_model'])
def test_controller_sota_rejects_before_prepare_or_any_model(harness, bad):
    from aka.execution.legacy.executor import LegacyExecutor
    h = harness
    descriptor = replace(h.request.descriptor, source_gpu=None) if bad == 'source' else h.request.descriptor
    target = {**TARGET, 'model': 'NVIDIA H100', 'evidence': 'Environment H100'} if bad == 'target_model' else TARGET
    sota = SotaStart(descriptor, target, 'H100' if bad == 'hardware' else 'L20D')
    driver = SchemaWorkflow(h.request.directory.parent / 'controller', planner=h.planner,
        executor=LegacyExecutor(), workspace=h.request.directory.parent / 'workspace', limits=RunLimits(max_episodes=1), sota=sota)
    campaign = SimpleNamespace(sandbox_hardware='L20D', tool_runtime=None)
    with pytest.raises(ValueError):
        driver.run(campaign)
    assert not h.scenario.calls and driver.state.data['prepared'] is None


def test_controller_frozen_sources_can_start_extract_after_original_loss(harness):
    from aka.workflow.schema.extract_sources import snapshot_sources
    h = harness
    frozen = snapshot_sources(h.request.directory.parent / 'controller-frozen', h.request.descriptor, target_gpu=TARGET)
    for path in (h.request.directory.parent / 'original').iterdir():
        path.unlink()
    result = h.extractor.extract(replace(h.request, source_snapshot=frozen.ref()))
    assert result.status == 'replace', result.diagnostics
    assert not (h.request.directory / 'sources').exists()
    assert state(h)['source'] == frozen.ref()


def test_source_reference_and_comparator_cannot_be_replaced_by_worker_claim(harness):
    h = harness
    def bad(i, c, files, r):
        doc = yaml.safe_load(files['root.yaml'])
        doc['CONTRACT']['Algorithm_requirement']['Golden_reference']['Tolerance']['y'] = 'atol=0.2, rtol=0, relative_to=reference, mode=elementwise'
        files['root.yaml'] = yaml.safe_dump(doc, sort_keys=False).encode()
    h.scenario.hooks[('P1', 'worker')] = bad
    result = h.extractor.extract(h.request)
    assert result.status == 'failed'
    assert all(role == 'worker' for _, _, role, _ in h.scenario.calls)
    assert any('comparator' in d['message'] for d in result.diagnostics)


@pytest.mark.parametrize('key', ['comparator:y', 'knowledge'])
def test_missing_per_output_conclusion_or_p6_knowledge_cannot_approve(harness, key):
    h = harness
    def missing(i, c, files, receipt):
        if key == 'knowledge':
            receipt['knowledge'] = []
        else:
            receipt['checks'].pop(key)
    h.scenario.hooks[('P6' if key == 'knowledge' else 'P1', 'reviewer')] = missing
    assert h.extractor.extract(h.request).status == 'failed'


@pytest.mark.parametrize('origin', ['extract', 'import'])
def test_optimize_inherits_explicit_extract_hardware_scope(harness, origin):
    from aka.workflow.schema.planner import PlannerRequest
    from aka.workflow.schema.language import CheckPolicy
    h = harness
    if origin == 'extract':
        result = h.extractor.extract(h.request)
        assert result.status == 'replace', result.diagnostics
        initial_ref = result.design_ref
    else:
        initial_ref = materials.publish_design(h.request.directory.parent / 'imported', FIXTURE,
            version=1, prepared_id='prepared-cpu', operation_id='imported-source-pair', materials_snapshot=h.planner.materials_snapshot,
            mode='import', policy=CheckPolicy(skip_hardware=True, style='advisory'))
    files = materials.read_design(initial_ref).checked.files.copy()
    root = yaml.safe_load(files['root.yaml'])
    root['DESIGN']['Math_reshape']['Rewrite'][0]['Math'] = 'y = x + 0\n'
    root['META_DATA']['Version'] += 1
    files['root.yaml'] = yaml.safe_dump(root, sort_keys=False).encode()
    h.scenario.optimize_files = files
    request = PlannerRequest('prepared-cpu', 'optimize-after-extract', 2, h.request.directory.parent / 'optimize',
        h.request.public_task, h.request.target_hardware, current_design=initial_ref)
    optimized = h.planner.optimize(request)
    assert optimized.status == 'replace', optimized.diagnostics
    artifact = materials.read_design(optimized.design_ref)
    assert artifact.manifest['check_policy'] == {'skip_hardware': True, 'style': 'advisory'}
    assert artifact.manifest['approval_refs'] == []


@pytest.mark.parametrize('bad', ['root_scalar', 'dispatch_scalar', 'decision_list', 'status_list'])
def test_malformed_role_output_is_a_bounded_bad_draft(harness, bad):
    h = harness
    def malformed(i, c, files, receipt):
        if bad == 'root_scalar':
            files['root.yaml'] = b'not-a-mapping\n'
        elif bad == 'dispatch_scalar':
            root = yaml.safe_load(files['root.yaml'])
            root['META_DATA']['Dispatch'] = 'not-a-mapping'
            files['root.yaml'] = yaml.safe_dump(root, sort_keys=False).encode()
        elif bad == 'decision_list':
            receipt['decision'] = []
        else:
            receipt['checks']['algorithm']['status'] = []
    h.scenario.hooks[('P1', 'reviewer' if bad == 'status_list' else 'worker')] = malformed
    result = h.extractor.extract(h.request)
    assert result.status == 'failed'
    assert result.calls <= 4


@pytest.mark.parametrize('location', ['captured', 'snapshot'])
def test_interrupted_role_replay_with_changed_raw_never_spends_repair_call(harness, monkeypatch, location):
    h = harness
    original = h.extractor._checked
    monkeypatch.setattr(h.extractor, '_checked', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('interrupted')))
    with pytest.raises(RuntimeError, match='interrupted'):
        h.extractor.extract(h.request)
    path = h.request.directory / 'stages/1' / ('worker/attempts/1/raw/design/root.yaml' if location == 'captured' else 'raw/root.yaml')
    path.chmod(0o600)
    path.write_bytes(path.read_bytes() + b'\n')
    monkeypatch.setattr(h.extractor, '_checked', original)
    with pytest.raises(RecoveryRequired, match='reauthor'):
        h.extractor.extract(h.request)
    assert len(h.scenario.calls) == 1
