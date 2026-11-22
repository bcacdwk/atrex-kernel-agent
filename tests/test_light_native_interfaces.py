"""CPU regression for explicit native seams; no Schema parser/model/GPU required."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from threading import RLock
from types import SimpleNamespace

import pytest

from aka.agent.provider import Agent
from aka.contracts.agent import AgentRunRequest
from aka.contracts.execution import CommandSpec, ExecutionContext, JobResult
from aka.contracts.problem import ProblemSnapshot
from aka.contracts.workspace import CandidateHandle, RevisionContent
from aka.execution.gateway.executor import GatewayExecutor
from aka.workflow.legacy.checkpoint import check_schema_mode, generate, read_schema_mode
from aka.workflow.light.long_horizon.models import EpisodeHandoff
from aka.workflow.light.long_horizon.store import CampaignStore
from aka.workflow.light.orchestration.supervisor_runtime import SupervisorRuntime
from schema_cpu_fixture import BASELINE, make_engine


def test_preparation_returns_final_task_without_v1_or_episode(tmp_path):
    campaign, _ = make_engine(tmp_path)
    campaign.problem_provider.snapshot = lambda problem: ProblemSnapshot(tmp_path, str(campaign.setup_count))
    public, snapshot = campaign.prepare_native()
    assert public.ready and snapshot.identity == '1'
    assert len(campaign.evaluations) == 1 and not campaign.agent.calls
    assert CampaignStore.checkpoint(campaign.workspace)[0].episodes == 0
    campaign.prepare_native()
    assert campaign.setup_count == 1
    campaign.finish_native('cpu-complete')
    campaign.finish_native('cpu-complete')
    assert campaign.finish_count == 1


def test_factory_preserves_native_verifier_settings_and_explicit_continuation(tmp_path):
    campaign, _ = make_engine(tmp_path)
    options = []
    campaign.acceptance_verifier = lambda **kw: options.append(kw) or SimpleNamespace()
    campaign.verify_repeats, campaign.verify_run_timeout, campaign.min_improvement_pct = 3, 19, 2.5
    engine = campaign.create_engine()
    assert engine.max_version == campaign.max_iters
    assert options[0]['repeats'] == 3 and options[0]['per_run_timeout'] == 19
    assert options[0]['min_improvement_pct'] == 2.5
    with pytest.raises(ValueError, match='max_iters=0'):
        campaign.create_engine(continuous=True)
    campaign.max_iters = 0
    engine = campaign.create_engine(continuous=True)
    assert engine.max_version is engine.max_episodes is None
    assert engine.token_budget == campaign.token_budget


def test_context_seed_and_resume_preserve_real_draft(tmp_path):
    seed = b'# explicit fixed-design seed\n'
    campaign, engine = make_engine(tmp_path, episode_context='Implement fixed design v1.',
        episode_seed=RevisionContent({'kernel.py': seed}))
    campaign.agent.fail = RuntimeError('CPU interruption')
    with pytest.raises(RuntimeError, match='CPU interruption'):
        engine.run(episode=1)
    assert campaign.agent.calls[0]['received'] == seed
    assert engine.run(episode=1) == 'episode-settled'
    assert campaign.agent.calls[1]['received'] == seed + b'\n# synthetic coder progress\n'
    assert campaign.agent.calls[1]['prompt'].endswith('Implement fixed design v1.')
    assert campaign.candidate_workspace.read_source(campaign.workspace).files['kernel.py'] == BASELINE
    assert engine.run(episode=1) == 'episode-settled'
    assert len(campaign.agent.calls) == 2
    engine.episode_context = 'different design'
    with pytest.raises(ValueError, match='retained binding'):
        engine.run(episode=1)


def test_audit_cannot_short_circuit_native_completion(tmp_path):
    campaign, engine = make_engine(tmp_path)
    audit = SimpleNamespace(completion_check=lambda *args: pytest.fail('native failure must win'))
    engine.candidate_audit = audit
    handle = CandidateHandle(1, 'a' * 40, 'episode', tmp_path)
    handoff = EpisodeHandoff('candidate_ready', candidate_commit='b' * 40)
    assert engine._completion_check(handle, tmp_path / 'missing-journal.json', handoff)


def test_mode_is_checked_before_off_or_checkpoint_and_cannot_take_over(tmp_path):
    workspace = tmp_path / 'on'
    check_schema_mode(workspace, enabled=True, root_ref='root-1', create=True)
    assert read_schema_mode(workspace) == {'schema_enabled': True, 'root_ref': 'root-1'}
    check_schema_mode(workspace, enabled=True, root_ref='root-1')
    with pytest.raises(ValueError, match='mode/root differs'):
        check_schema_mode(workspace)
    with pytest.raises(ValueError, match='export/import'):
        generate(workspace, candidate_workspace=None)
    existing = tmp_path / 'existing'
    existing.mkdir()
    (existing / 'kernel.py').write_text('user source')
    with pytest.raises(ValueError, match='take over'):
        check_schema_mode(existing, enabled=True, root_ref='root-1', create=True)
    off = tmp_path / 'off'
    check_schema_mode(off)
    assert not off.exists()


def test_schema_infrastructure_outage_is_returned_without_repeat(tmp_path):
    from aka.workflow.light.orchestration.infrastructure_retry import (
        InfrastructureUnavailable, bounded_infrastructure, retry_infrastructure)
    calls = []
    def unavailable():
        calls.append(1)
        raise InfrastructureUnavailable('unknown original job')
    with bounded_infrastructure(), pytest.raises(InfrastructureUnavailable):
        retry_infrastructure(tmp_path, 'one-owned-attempt', unavailable)
    assert calls == [1]


def test_agent_scoped_admission_result_denial_and_stale_session(tmp_path):
    calls, events = [], []
    agent = Agent(process_runner=lambda *args, **kw: calls.append(args) or ('', '', 0, False),
        environment={'HOME': str(tmp_path), 'PATH': '/usr/bin:/bin', 'ANTHROPIC_MODEL': 'qwen3.8-max[1m]'})
    def admit(**request):
        assert request['backend'] == 'claude'
        assert request['environment']['ANTHROPIC_MODEL'] == 'qwen3.8-max[1m]'
        events.append('admit')
        return 'permit'
    def observe(permit, *, event, value):
        assert permit == 'permit'
        events.append(event)
    with agent.invocation_scope(admit=admit, observe=observe):
        runtime = agent.runtime('claude')
        runtime.run(AgentRunRequest(tmp_path, 'CPU fake', 5))
        session = agent.open_session('claude', tmp_path)
        session.invoke('CPU fake', environment={'ATREX_AGENT_TIMEOUT_SECONDS': '5'})
    with pytest.raises(RuntimeError, match='scope has been closed'):
        session.invoke('must not start', resume=True)
    session.close()
    with pytest.raises(RuntimeError, match='scope has been closed'):
        runtime.run(AgentRunRequest(tmp_path, 'stale runtime', 5))
    assert len(calls) == 2 and events == ['admit', 'result', 'admit', 'result']
    def deny(**request):
        raise RuntimeError('root denied')
    with agent.invocation_scope(admit=deny, observe=observe), pytest.raises(RuntimeError, match='root denied'):
        agent.runtime('claude').run(AgentRunRequest(tmp_path, 'denied', 5))
    assert len(calls) == 2
    agent.close()


def test_process_observations_are_optional_and_cumulative(tmp_path):
    from aka.agent.execution.process import run_bounded
    seen = []
    command = [sys.executable, '-c', "import time; print('cpu output', flush=True); time.sleep(.25)"]
    result = run_bounded(command, tmp_path, 3, on_observation=seen.append)
    assert result[2] == 0 and 'cpu output' in result[0]
    assert seen and b'cpu output' in seen[-1]['stdout']
    assert seen[-1]['active_seconds'] >= 0


def test_gateway_scope_follows_bind_observes_result_and_denies_replay(tmp_path, monkeypatch):
    from aka.contracts.execution_environment import EnvironmentQuery
    from aka.execution.environment_query import _snapshot, match_environment
    fields = json.loads((Path(__file__).parent / 'fixtures/agate_b41_environment.json').read_text())
    environment = _snapshot('gateway', 'cpu-fixture', EnvironmentQuery('L20D'),
        fields['detail'], fields['capabilities'], [], GatewayExecutor.capabilities.operations)
    support = match_environment(EnvironmentQuery('L20D', 'dev'), environment)
    executor = GatewayExecutor()
    events = []
    def admit(**kw):
        events.append('admit')
        return 1
    def observe(permit, *, event, value):
        assert permit == 1
        events.append(event)
    def run_owned(*args, **kw):
        result = JobResult(0, exit_code=0)
        kw['on_observation'](result)
        return result
    monkeypatch.setattr('aka.execution.outcomes.run_owned', run_owned)
    command = CommandSpec(tmp_path, ('true',), hardware='L20D', operation='dev', timeout=5)
    with executor.dispatch_scope(admit=admit, observe=observe):
        bound = executor.bind(ExecutionContext({}, 'L20D', 5))
        monkeypatch.setattr(bound, '_environment_backend', lambda *args: SimpleNamespace(check=lambda *args, **kw: support))
        monkeypatch.setattr(bound, '_job_transport', lambda: SimpleNamespace(environment={}, endpoint='', profile=''))
        assert bound.run(command).exit_code == 0
        with pytest.raises(RuntimeError, match='replay'):
            bound.replay_job(command, idempotency_key='same')
    assert events == ['admit', 'observation', 'result']
    with pytest.raises(RuntimeError, match='scope has been closed'):
        bound.run(command)


def test_runtime_draft_is_stopped_source_not_sealed_candidate(tmp_path):
    campaign, _ = make_engine(tmp_path)
    runtime = SupervisorRuntime.__new__(SupervisorRuntime)
    runtime.root, runtime.audit_root = tmp_path, None
    runtime.workspace_provider = campaign.candidate_workspace
    runtime.lock, runtime.capabilities = RLock(), {}
    directory = runtime.episode_journal_path(1).parent / 'agent'
    draft = directory / 'workspace'
    draft.mkdir(parents=True)
    (draft / 'kernel.py').write_bytes(b'# real draft B\n')
    (directory / 'state.json').write_text('{}')
    assert runtime.read_episode_draft(1).files['kernel.py'] == b'# real draft B\n'
    runtime.capabilities['live'] = SimpleNamespace(journal=SimpleNamespace(
        service=SimpleNamespace(path=runtime.episode_journal_path(1))))
    with pytest.raises(RuntimeError, match='live capability'):
        runtime.read_episode_draft(1)


def test_additional_unknown_qualification_cannot_reach_verifier(tmp_path, monkeypatch):
    campaign, engine = make_engine(tmp_path,
        candidate_qualification=lambda **kw: {"qualified": None, "reason": "unknown"})
    events = []
    engine._completion_check = lambda *args: events.append('native-completion') or ''
    campaign.candidate_workspace.validate_candidate = lambda *args: ('', ['kernel.py'])
    monkeypatch.setattr('aka.workflow.light.long_horizon.main_adapter.candidate_policy_violations',
        lambda *args, **kw: [])
    monkeypatch.setattr('aka.workflow.light.long_horizon.main_adapter.candidate_is_gluon', lambda *args: False)
    active = {}
    store = SimpleNamespace(save_active=lambda value: events.append(dict(value)))
    verifier = SimpleNamespace(verify=lambda *args, **kw: pytest.fail('unqualified candidate cannot run ABBA'))
    result = engine._assess_terminal_handoff(store, active,
        CandidateHandle(1, 'a' * 40, 'episode', tmp_path),
        EpisodeHandoff('candidate_ready', candidate_commit='b' * 40),
        memory_version=1, conversion_pending=False, verifier=verifier)
    assert result[0] == 'additional candidate qualification is incomplete'
    assert result[2] is None and result[3] is False
    assert events[0] == 'native-completion' and events[1]['candidate_qualification']['qualified'] is None


def test_off_public_workflows_reject_on_before_loading_task(tmp_path):
    from aka.workflow.legacy.assembly import LegacyWorkflow
    from aka.workflow.light.provider import LightWorkflow
    check_schema_mode(tmp_path / 'on', enabled=True, root_ref='root', create=True)
    request = SimpleNamespace(workspace=tmp_path / 'on')
    legacy = LegacyWorkflow(SimpleNamespace(agent=None,
        problem=SimpleNamespace(load=lambda *args, **kw: pytest.fail('must check mode first'))))
    with pytest.raises(ValueError, match='mode/root differs'):
        legacy._load_problem(request)
    light = LightWorkflow.__new__(LightWorkflow)
    light._closed, light.driver = False, None
    with pytest.raises(ValueError, match='mode/root differs'):
        light._load_problem(request)


def test_original_experiment_records_and_sources_are_read_without_execution(tmp_path):
    records = (object(), object())
    artifact = object()
    source = RevisionContent({'kernel.py': b'# measured A'})
    @contextmanager
    def open_records(**kwargs):
        assert kwargs['workspace'] == tmp_path
        yield SimpleNamespace(iter_records=lambda: iter(records),
                              read_source=lambda selected: source if selected is artifact else None)
    runtime = SupervisorRuntime.__new__(SupervisorRuntime)
    runtime.config = SimpleNamespace(workspace=tmp_path)
    runtime.executor = runtime.evaluation_input = None
    runtime._record_services = SimpleNamespace(experiments=SimpleNamespace(open=open_records),
        evaluator=None, command_evaluator=None)
    assert runtime.read_experiment_records() == records
    assert runtime.read_experiment_source(artifact) is source


def test_continuous_exact_episodes_cross_eight_without_resetting_native_state(tmp_path):
    campaign, _ = make_engine(tmp_path)
    campaign.max_iters = 0
    engine = campaign.create_engine(continuous=True)
    for number in range(1, 10):
        assert engine.run(episode=number) == 'episode-settled'
    # Reconstruct the engine against the same Campaign and native checkpoint.
    assert campaign.create_engine(continuous=True).run(episode=10) == 'episode-settled'
    state, active = CampaignStore.checkpoint(campaign.workspace)
    assert active is None and state.episodes == 10 and state.tokens == 70
    assert len(campaign.agent.calls) == 10 and campaign.setup_count == 1
    assert len(campaign.evaluations) == 1


@pytest.mark.parametrize('boundary', ['materialize', 'draft'])
def test_preparing_crash_replays_same_native_episode_and_seed(tmp_path, monkeypatch, boundary):
    seed = RevisionContent({'kernel.py': b'# fixed seed'})
    campaign, engine = make_engine(tmp_path, episode_context='one fixed design', episode_seed=seed)
    calls = []
    owner = campaign.candidate_workspace if boundary == 'materialize' else campaign
    method = 'materialize' if boundary == 'materialize' else 'register_runtime_episode'
    original = getattr(owner, method)
    def interrupted(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 1:
            raise RuntimeError('preparation interruption')
        return original(*args, **kwargs)
    monkeypatch.setattr(owner, method, interrupted)
    with pytest.raises(RuntimeError, match='preparation interruption'):
        engine.run(episode=1)
    state, active = CampaignStore.checkpoint(campaign.workspace)
    assert state.episodes == 0 and active['phase'] == 'preparing' and not campaign.agent.calls
    assert engine.run(episode=1) == 'episode-settled'
    assert calls[0] == calls[1]
    assert campaign.agent.calls[0]['received'] == seed.files['kernel.py']
    assert len(campaign.agent.calls) == 1
