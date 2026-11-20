"""Synthetic CPU services with real native preparation, Store, drafts and Git.

No Task/GPU/production qualification is claimed by this fixture. The fake
evaluator executes a small correct CPU baseline; timings are synthetic.
"""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

from aka.contracts.agent import AgentRuntimeCapabilities, NativeInvocation, TokenUsage
from aka.contracts.content import PublicInputContent, SourceContent, WorkspaceFile
from aka.contracts.filesystem import FilesystemAccess
from aka.contracts.problem import EvaluationInput, InitialSource
from aka.resources.plugins import PluginRegistry
from aka.task.candidate_workspace.provider import GitCandidateWorkspace
from aka.workflow.light.campaign import ManagedCampaign
from aka.workflow.light.long_horizon.campaign import LongHorizonCampaign
from aka.workflow.light.long_horizon.session import LongSessionRunner
from aka.workflow.light.orchestration.episode_workspace import EpisodeWorkspace
from aka.workflow.light.orchestration.constants import TEST_RESULT_PREFIX


BASELINE = b"""import math
def softmax(x, out):
    values = [math.exp(value - max(x)) for value in x]
    out[:] = [value / sum(values) for value in values]
    return out
"""


class FakeAgent:
    def __init__(self):
        self.calls = []
        self.fail = None

    def open_session(self, agent_cli, workspace, *, session_id, environment, **kwargs):
        agent = self
        class Session:
            supports_resume = False
            def invoke(self, prompt, **kwargs):
                received = (workspace / "kernel.py").read_bytes()
                agent.calls.append({"received": received, "prompt": prompt, "workspace": workspace})
                (workspace / "kernel.py").write_bytes(received + b"\n# synthetic coder progress\n")
                if agent.fail is not None:
                    failure, agent.fail = agent.fail, None
                    raise failure
                # Native invalid_handoff is deliberate: partial work is retained,
                # without pretending that a fake report qualifies for promotion.
                return NativeInvocation(1, False, session_id, "", "synthetic partial",
                    TokenUsage(4, 3, 0, 0, 7, "exact"), (),
                    AgentRuntimeCapabilities(True, False, False), (), False)
            def close(self):
                pass
        session = Session()
        session.environment = environment
        return session


class FakeRuntime:
    def __init__(self, campaign):
        self.campaign = campaign
        self.plugins = campaign._module_resources
        self.views = {}

    def episode_journal_path(self, episode):
        # This fake Agent never submitted a terminal runtime report.
        return self.campaign.workspace / ".atrex_long_horizon" / "private-reports" / f"e{episode:04d}.json"

    @contextmanager
    def session(self, workspace, environment, **kwargs):
        try:
            yield SimpleNamespace(environment=environment, filesystem_access=kwargs.get("access"))
        finally:
            self.views[workspace].publish()


class CpuCampaign(ManagedCampaign):
    def __init__(self, root):
        super().__init__(name="synthetic_cpu", kernel_demo=str(root / "input.py"),
                         platform="cpu", framework="triton", framework_baseline="never",
                         convert_after=0, target_util=0)
        self.request = SimpleNamespace(workspace=root / "native")
        # Select native command semantics; execution below is explicitly a fake
        # CPU service, not an installed Atrex-Bench qualification.
        self.atrex_bench_root = str(root / "synthetic-evaluator")
        self.candidate_workspace = GitCandidateWorkspace()
        self.agent = FakeAgent()
        self.module_options = SimpleNamespace(file_system="single_file", goal_episodes=False,
            episode_timeout_seconds=60, output_limit_recovery_retries=0, generate_memory=False, search=None)
        self.tool_runtime = SimpleNamespace(wiki=False, research_records=False)
        self._module_resources = PluginRegistry(resources=(), agent_skills=())
        initial = InitialSource("kernel", root / "input.py", (WorkspaceFile("kernel.py", text=BASELINE.decode()),))
        evaluation = EvaluationInput("synthetic-cpu", root, (),
            harness=WorkspaceFile("test_kernel.py", text="# synthetic CPU evaluator\n"))
        self.problem = SimpleNamespace(initial_source=initial, evaluation=evaluation)
        self.source_provider = SimpleNamespace(prepare=lambda request: SourceContent(initial.files))
        inputs = SimpleNamespace(prepare_public=lambda: PublicInputContent(True),
            operator_files=lambda: (), generated_digest="", assert_private=lambda: None)
        self.problem_provider = SimpleNamespace(content=lambda *args, **kwargs: inputs,
                                               runtime_files=lambda problem: ())
        self.episode_evidence_reader = SimpleNamespace(read_episode_receipts=lambda *args, **kwargs: ())
        self.episode_evidence_interpreter = SimpleNamespace(interpret_episode_receipts=lambda *args, **kwargs: ())
        self.setup_count = self.finish_count = 0
        self.evaluations = []
        self.evaluation_failure = None
        self._supervisor_runtime = FakeRuntime(self)

    def setup_baseline(self):
        self.setup_count += 1
        return super().setup_baseline()

    def _finish(self, reason):
        self.finish_count += 1
        return super()._finish(reason)

    def start_runtime(self):
        pass

    def acceptance_verifier(self, **kwargs):
        return SimpleNamespace()

    def measure_for_acceptance(self, workspace, argv, **kwargs):
        source = (workspace / "kernel.py").read_bytes()
        self.evaluations.append((tuple(argv), source))
        if self.evaluation_failure is not None:
            raise self.evaluation_failure
        namespace = {}
        exec(compile(source, "synthetic_baseline", "exec"), namespace)
        out = [float("nan")] * 3
        assert namespace["softmax"]([1.0, 1.0, 1.0], out) is out
        assert out == [1 / 3] * 3
        result = {"all_pass": True, "latency_us_geomean": 10.0,
                  "latency_us_by_shape": {"synthetic": 10.0}}
        return {"exit_code": 0, "stdout": TEST_RESULT_PREFIX + json.dumps(result), "stderr": ""}

    def register_runtime_episode(self, worktree, *, episode, **kwargs):
        directory = self.workspace / ".atrex_long_horizon" / "drafts" / f"e{episode:04d}"
        view = EpisodeWorkspace(worktree, directory, resources_registry=self._module_resources,
                                workspace_provider=self.candidate_workspace)
        draft = view.prepare()
        self._supervisor_runtime.views[draft] = view
        return draft

    def agent_access(self, workspace, **kwargs):
        return FilesystemAccess()


def make_engine(root, **options):
    campaign = CpuCampaign(root)
    runner = LongSessionRunner(agent=campaign.agent, usage_policy=lambda turn: turn.terminal_usage.total_tokens)
    engine = LongHorizonCampaign(campaign, max_version=options.pop("max_version", 8),
        token_budget=options.pop("token_budget", 70), verifier=SimpleNamespace(),
        session_runner=runner, handoff_resumes=0, **options)
    return campaign, engine


class Owner:
    """Synthetic owner: a callback probe, deliberately not a second budget ledger."""
    def __init__(self, root):
        self.root = root
        self.events = []
        self.denied = set()
        self.terminals = {}
        self.settlement_failure = None

    def admit(self, *, phase, episode, workspace, active):
        self.events.append(("admit", phase, episode))
        # Prove native location can be recorded before V0/first result. A5 adds
        # fixed implementation/operation/owner identities to the actual anchor.
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "native-locator.json").write_text(json.dumps({
            "workspace": str(workspace), "episode": episode, "phase": phase}))
        return phase not in self.denied

    def settled(self, *, episode, workspace, attempt):
        assert not (workspace / ".atrex_long_horizon/active_episode.json").exists()
        assert json.loads((workspace / f".atrex_long_horizon/episodes/e{episode:04d}/attempt.json").read_text()) == attempt
        self.events.append(("settled", episode))
        if self.settlement_failure is not None:
            failure, self.settlement_failure = self.settlement_failure, None
            raise failure
        self.terminals.setdefault(episode, attempt)
