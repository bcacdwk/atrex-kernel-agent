"""Stopped original-location checkpoint, composing each module's public snapshot."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import tempfile

from aka.bootstrap.continuation import read as read_selection, restore as restore_selection, digest
from aka.contracts.agent import AgentCheckpointSnapshot
from aka.contracts.problem import ProblemSnapshot
from aka.contracts.workflow_export import CheckpointInfo
from aka.contracts.workflow_launch import WorkflowLaunch, encode_launch
from aka.contracts.workspace import CandidateWorkspaceSnapshot
from .export import document, RetainedWorkflowExport
from .public_state import campaign_lease, writer_active

SCHEMA = 'aka.workflow.checkpoint.v1'
CONTEXT = '.atrex_long_horizon/checkpoint-context.json'
CAPTURED_FACTS = '.atrex_long_horizon/checkpoint-facts.json'
SCHEMA_MODE = '.atrex_long_horizon/schema-mode.json'


def _decode_schema_mode(raw):
    if raw is None:
        return None
    value = json.loads(raw)
    if (not isinstance(value, dict) or set(value) != {'schema_enabled', 'root_ref'}
            or type(value['schema_enabled']) is not bool
            or (value['schema_enabled'] and (not isinstance(value['root_ref'], str) or not value['root_ref']))
            or (not value['schema_enabled'] and value['root_ref'] is not None)):
        raise ValueError('invalid native Schema mode marker')
    return value


def read_schema_mode(workspace):
    """Read only the native marker; never import the optional Schema package."""
    from aka.core.filesystem import read_regular_bytes
    try:
        raw = read_regular_bytes(Path(workspace) / SCHEMA_MODE, limit=16 * 1024 + 1)
    except FileNotFoundError:
        return None
    if len(raw) > 16 * 1024:
        raise ValueError('native Schema mode marker exceeds its size limit')
    return _decode_schema_mode(raw)


def check_schema_mode(workspace, *, enabled=False, root_ref=None, create=False):
    """Check mode before Task loading; only an explicit on owner creates it."""
    workspace = Path(workspace)
    expected = {'schema_enabled': enabled, 'root_ref': root_ref}
    _decode_schema_mode(json.dumps(expected))
    previous = read_schema_mode(workspace)
    if previous is not None and previous != expected:
        raise ValueError('workspace Schema mode/root differs; resume its original controller')
    if previous is None and enabled:
        if workspace.exists() and any(workspace.iterdir()):
            raise ValueError('Schema cannot take over an existing unmarked workspace')
        if create:
            from aka.core.filesystem import durable_write_json
            durable_write_json(workspace / SCHEMA_MODE, expected)
    return previous


def _reject_schema_checkpoint(workspace):
    mode = read_schema_mode(workspace)
    if mode is not None and mode['schema_enabled']:
        raise ValueError('Schema checkpoint export/import is unsupported; resume the original root in place')


def record_context(campaign):
    payload = read_selection(os.environ)
    if payload is None:
        return  # A direct in-process caller has not supplied recoverable selection.
    snapshot = campaign.problem_provider.snapshot(campaign.problem)
    request = replace(campaign.request, problem=replace(campaign.request.problem,
                      snapshot_directory=snapshot.directory))
    argv = encode_launch(WorkflowLaunch(request, request.workspace.parent))
    path = campaign.workspace / CONTEXT
    previous = document(path)
    if previous and (previous['selection'] != payload or previous['argv'] != list(argv)):
        raise ValueError('campaign request/selection differs from its checkpoint context')
    value = {'schema':1, 'selection':payload, 'argv':argv,
             'problem':{'directory':str(snapshot.directory), 'identity':snapshot.identity},
             'resources':campaign.plugin_registry.resource_snapshot()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True)+'\n')
    temporary.replace(path)


@contextmanager
def _services(payload):
    from aka.bootstrap.recovery_metadata import restored_services
    with restored_services(payload, ('problem', 'candidate', 'agent')) as services:
        from aka.resources.plugins import PluginRegistry
        yield (*services, PluginRegistry())


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _pack(root, metadata):
    entries = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root).as_posix()
        mode = stat.S_IMODE(path.lstat().st_mode)
        if path.is_symlink():
            entries.append({'path':relative,'kind':'symlink','target':str(path.readlink()),'mode':mode})
        elif path.is_file():
            entries.append({'path':relative,'kind':'file','sha256':_hash(path.read_bytes()),'mode':mode})
        elif path.is_dir():
            entries.append({'path':relative,'kind':'directory','mode':mode})
        else:
            raise ValueError('unsupported checkpoint content')
    metadata = {**metadata,'entries':entries}
    encoded = (json.dumps(metadata, sort_keys=True)+'\n').encode()
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as archive:
        header = tarfile.TarInfo('checkpoint.json'); header.size=len(encoded); header.mode=0o600
        archive.addfile(header, io.BytesIO(encoded))
        for row in entries:
            archive.add(root/row['path'], arcname=row['path'], recursive=False)
    return output.getvalue()


def _unpack(content, root):
    with tarfile.open(fileobj=io.BytesIO(content), mode='r:gz') as archive:
        members = archive.getmembers()
        by_name = {m.name:m for m in members}
        if len(by_name) != len(members) or 'checkpoint.json' not in by_name or not by_name['checkpoint.json'].isfile():
            raise ValueError('invalid checkpoint members')
        metadata = json.load(archive.extractfile('checkpoint.json'))
        if metadata.get('schema') != SCHEMA:
            raise ValueError('unsupported Workflow checkpoint')
        rows = metadata['entries']
        if {r['path'] for r in rows} != set(by_name)-{'checkpoint.json'} or len(rows) != len(members)-1:
            raise ValueError('checkpoint membership differs')
        links = {r['path'] for r in rows if r['kind']=='symlink'}
        for row in rows:
            path = PurePosixPath(row['path'])
            if path.is_absolute() or '..' in path.parts or not path.parts or str(path) != row['path']:
                raise ValueError('unsafe checkpoint path')
            if any(str(p) in links for p in path.parents):
                raise ValueError('checkpoint writes through a symlink')
            member = by_name[row['path']]
            if member.mode != row['mode']:
                raise ValueError('checkpoint mode changed')
            target = root/row['path']
            if row['kind']=='file':
                if not member.isfile(): raise ValueError('checkpoint entry type changed')
                data = archive.extractfile(member).read()
                if _hash(data)!=row['sha256']: raise ValueError('checkpoint content digest changed')
                target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data);target.chmod(member.mode)
            elif row['kind']=='directory':
                if not member.isdir(): raise ValueError('checkpoint entry type changed')
                target.mkdir(parents=True,exist_ok=True)
            elif row['kind']=='symlink':
                if not member.issym() or member.linkname != row['target']: raise ValueError('checkpoint link changed')
            else: raise ValueError('unsupported checkpoint entry')
        for row in rows:
            if row['kind']=='symlink':
                target=root/row['path'];target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(row['target'])
        # Apply directory permissions after children; retained trees may be read-only.
        for row in sorted(rows, key=lambda row:len(PurePosixPath(row['path']).parts), reverse=True):
            if row['kind']=='directory':
                (root/row['path']).chmod(row['mode'])
    return metadata


def _info(metadata, content):
    return CheckpointInfo(metadata['campaign_id'], Path(metadata['workspace']), _hash(content),
                          metadata['tokens'], metadata['episodes'], metadata['incumbent_version'],
                          digest(metadata['context']['selection']))


def generate(workspace, *, candidate_workspace):
    workspace=Path(workspace).resolve()
    _reject_schema_checkpoint(workspace)
    with campaign_lease(workspace):
        stopped=document(workspace/'.atrex_long_horizon/writer-state.json')
        # An idle lock does not turn a hard-crashed writer into a recorded clean stop.
        if stopped and stopped.get('state') == 'running':
            raise ValueError(
                'checkpoint cannot use stale running writer state after an unclean stop; '
                'resume the campaign and reach a recorded clean stop'
            )
        if not stopped or stopped.get('state')!='stopped':
            raise ValueError('checkpoint requires a recorded clean stop at an episode boundary')
        if document(workspace/'.atrex_long_horizon/active_episode.json'):
            raise ValueError('checkpoint requires a finished episode; request_pause and wait for stop')
        context=document(workspace/CONTEXT)
        if context is None: raise ValueError('campaign has no frozen launch context')
        facts=RetainedWorkflowExport(candidate_workspace).read(workspace)
        if facts.incumbent is None: raise ValueError('checkpoint requires a verified incumbent')
        if any(type(value) is not int or value < 0 for value in
               (facts.progress.observed_tokens, facts.progress.completed_episodes)):
            raise ValueError('checkpoint requires complete budget counters')
        captured = {'campaign_id':facts.campaign.id, 'tokens':facts.progress.observed_tokens,
                    'episodes':facts.progress.completed_episodes, 'incumbent_version':facts.incumbent.version,
                    'incumbent_revision':facts.incumbent.revision}
        (workspace/CAPTURED_FACTS).write_text(json.dumps(captured,sort_keys=True)+'\n')
        with _services(context['selection']) as (problem,candidate,agent,registry):
            registry.check_resource_snapshot(context['resources'])
            loaded=problem.validate_snapshot(ProblemSnapshot(Path(context['problem']['directory']),context['problem']['identity']))
            with tempfile.TemporaryDirectory(prefix='aka-checkpoint-') as raw:
                root=Path(raw).resolve()
                ps=problem.export_snapshot(loaded,root/'problem')
                ws=candidate.export_checkpoint(workspace,root/'workspace')
                ag=agent.export_checkpoint(workspace,root/'agent')
                metadata={'schema':SCHEMA,'workspace':str(workspace),'campaign_id':facts.campaign.id,
                          'context':context,'problem_identity':ps.identity,'workspace_identity':ws.identity,
                          'agent_identity':ag.identity,'tokens':facts.progress.observed_tokens,
                          'episodes':facts.progress.completed_episodes,'incumbent_version':facts.incumbent.version,
                          'incumbent_revision':facts.incumbent.revision}
                return _pack(root,metadata)


def _validate(root, metadata, expected_workspace, expected_campaign_id, services):
    _reject_schema_checkpoint(expected_workspace)
    if any(type(metadata.get(key)) is not int or metadata[key] < 0 for key in ('tokens','episodes','incumbent_version')):
        raise ValueError('invalid checkpoint counters')
    workspace=Path(metadata['workspace'])
    if workspace != workspace.resolve() or not workspace.is_absolute():
        raise ValueError('checkpoint requires its original absolute workspace path')
    if workspace != Path(expected_workspace).absolute() or metadata['campaign_id'] != expected_campaign_id:
        raise ValueError('checkpoint belongs to another campaign or original workspace')
    problem,candidate,agent,registry=services
    context=metadata['context']
    registry.check_resource_snapshot(context['resources'])
    ps=ProblemSnapshot(root/'problem',metadata['problem_identity'])
    ws=CandidateWorkspaceSnapshot(root/'workspace',metadata['workspace_identity'])
    ag=AgentCheckpointSnapshot(root/'agent',metadata['agent_identity'])
    target=Path(context['problem']['directory'])
    problem.validate_snapshot(ps,destination=target)
    state=candidate.validate_checkpoint(ws,check_destinations=True)
    for entry in metadata['entries']:
        if entry['path'].endswith('/' + SCHEMA_MODE):
            if (_decode_schema_mode((root / entry['path']).read_bytes()) or {}).get('schema_enabled'):
                raise ValueError('Schema checkpoint import is unsupported')
    if state.workspace != workspace or state.candidates:
        raise ValueError('checkpoint workspace/state does not match stopped mode')
    agent.validate_checkpoint(ag,check_destinations=True)
    owned_paths=('.atrex_long_horizon/public-identity.json', '.atrex_long_horizon/state.json',
                 CONTEXT, '.atrex_long_horizon/writer-state.json',
                 f"memory/v{metadata['incumbent_version']}.json", CAPTURED_FACTS)
    owned=candidate.checkpoint_files(ws,owned_paths)
    identity=json.loads(owned[owned_paths[0]])
    counters=json.loads(owned[owned_paths[1]])
    saved_context=json.loads(owned[CONTEXT])
    stopped=json.loads(owned[owned_paths[3]])
    incumbent=json.loads(owned[owned_paths[4]])
    captured=json.loads(owned[CAPTURED_FACTS])
    if (captured != {key:metadata[key] for key in
                     ('campaign_id','tokens','episodes','incumbent_version','incumbent_revision')}
            or incumbent.get('git_commit_hash') != metadata['incumbent_revision']
            or identity.get('campaign_id') != metadata['campaign_id']
            or counters.get('tokens') != metadata['tokens']
            or counters.get('episodes') != metadata['episodes']
            or saved_context != context or stopped.get('state') != 'stopped'
            or incumbent.get('correctness',{}).get('status') != 'PASS'
            or incumbent.get('quality_gate',{}).get('result') != 'PASS'
            or incumbent.get('performance',{}).get('measurement_status') != 'complete'):
        raise ValueError('checkpoint metadata differs from retained Workflow facts')
    from aka.contracts.workflow_launch import decode_launch
    request=decode_launch(tuple(context['argv'])).request
    if request.workspace != workspace or request.problem.snapshot_directory != target:
        raise ValueError('checkpoint launch target differs from retained destinations')
    if context['problem']['identity'] != metadata['problem_identity']:
        raise ValueError('checkpoint Task identity differs from frozen context')
    return ps,ws,ag,target


def validate(content, *, workspace, campaign_id):
    _reject_schema_checkpoint(workspace)
    with tempfile.TemporaryDirectory(prefix='aka-checkpoint-validate-') as raw:
        root=Path(raw).resolve(); metadata=_unpack(content,root)
        with _services(metadata['context']['selection']) as services:
            if writer_active(workspace):
                raise ValueError('campaign has an active writer')
            _validate(root,metadata,workspace,campaign_id,services)
        return _info(metadata,content)


def restore(content, *, workspace, campaign_id):
    _reject_schema_checkpoint(workspace)
    with tempfile.TemporaryDirectory(prefix='aka-checkpoint-restore-') as raw:
        root=Path(raw).resolve();metadata=_unpack(content,root)
        with _services(metadata['context']['selection']) as services:
            _validate(root,metadata,workspace,campaign_id,services)
            with campaign_lease(workspace):
                ps,ws,ag,target=_validate(root,metadata,workspace,campaign_id,services)
                problem,candidate,agent,_=services
                # All modules and all destinations have passed their no-write check.
                problem.restore_snapshot(ps,target)
                candidate.restore_checkpoint(ws)
                agent.restore_checkpoint(ag)
        return _info(metadata,content)


def resume(workspace, *, campaign_id, candidate_workspace):
    _reject_schema_checkpoint(workspace)
    from aka.bootstrap.host import run_selection
    from aka.contracts.startup import Invocation
    workspace=Path(workspace).resolve()
    facts=RetainedWorkflowExport(candidate_workspace).read(workspace)
    if facts.campaign.id!=campaign_id:
        raise ValueError('cannot resume a different campaign')
    context=document(workspace/CONTEXT)
    if not context:
        raise ValueError('campaign has no frozen continuation')
    if facts.progress.writer_active or document(workspace/'.atrex_long_horizon/active_episode.json'):
        raise ValueError('checkpoint resume requires a stopped episode boundary')
    if facts.incumbent is None or any(type(value) is not int or value < 0 for value in
                                     (facts.progress.observed_tokens,facts.progress.completed_episodes)):
        raise ValueError('checkpoint resume requires verified state and budget counters')
    selection=restore_selection(context['selection'])
    return run_selection(selection,Invocation(tuple(context['argv'])))
