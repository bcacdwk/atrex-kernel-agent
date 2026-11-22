"""CPU substitutions below the real Bootstrap/Light/Schema assembly seams.

The selected Agent and native session are real. Only its process is synthetic;
CpuCampaign retains native preparation, Git, Episode, draft and budget owners.
No fake controller, Planner, plugin, host or startup is installed here.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from aka.agent.provider import Agent
from aka.contracts.agent import AgentCapabilities
from aka.contracts.content import PublicInputContent, WorkspaceFile
from aka.contracts.problem import AgentProblem, LoadedProblem, ProblemSnapshot
from schema_cpu_fixture import BASELINE, CpuCampaign


FIXTURE = Path(__file__).parent / "fixtures/schema_language/minimal"


class CpuAgent(Agent):
    @property
    def capabilities(self):
        # Mount declarations are inspected; this does not qualify real bwrap.
        return AgentCapabilities(filesystem_isolation=True, tool_runtime=True)


def install_native_cpu(monkeypatch, directory):
    directory.mkdir(parents=True, exist_ok=True)
    campaigns, agents, calls = [], [], []
    task_directory = directory / "task"
    task_directory.mkdir(exist_ok=True)
    (task_directory / "reference.py").write_bytes(BASELINE)
    (directory / "input.py").write_bytes(BASELINE)

    def process(command, *, cwd, timeout, env):
        workspace = Path(cwd)
        assert command[0] == "claude"
        assert env["ANTHROPIC_MODEL"] == "qwen3.8-max[1m]"
        assert timeout is not None and 0 < timeout <= 3600
        request_file = workspace / "inputs/task/request.json"
        if request_file.is_file():
            request = json.loads(request_file.read_text())
            kind = "optimize" if (workspace / "inputs/current-design").is_dir() else "design"
            calls.append({"kind": kind, "request": request, "workspace": workspace})
            version = request["version"]
            decision = {"status": "replace", "rationale": "Synthetic scalar rewrite for public entry testing",
                "hypothesis_key": f"public-cpu-{version}",
                "changed_execution_decisions": ["Use the next equivalent scalar expression"],
                "changes": [{"path": "root.DESIGN.Math_reshape", "before": "previous expression",
                             "after": f"x + ({version} - {version})", "reason": "CPU fixture"}],
                "evidence": [{"source": "task/public.md", "start": 1, "end": 1,
                              "claim": "Synthetic public task only"}],
                "next_actions": ["Implement this expression and submit native evidence"]}
            (workspace / "output/decision.json").write_text(json.dumps(decision))
            for source in FIXTURE.iterdir():
                raw = source.read_bytes()
                if source.name == "root.yaml":
                    raw = raw.replace(b"          y = x\n  Work_schedule:",
                        f"          y = x + ({version} - {version})\n  Work_schedule:".encode())
                (workspace / "output/design" / source.name).write_bytes(raw)
            text, code = "synthetic design written", 0
        else:
            candidate = workspace / "kernel.py"
            received = candidate.read_bytes()
            calls.append({"kind": "implement", "received": received, "workspace": workspace})
            candidate.write_bytes(received + b"\n# synthetic public-entry coder progress\n")
            # Deliberately incomplete native handoff: no synthetic promotion.
            text, code = "synthetic incomplete native handoff", 1
        return json.dumps({"type": "result", "result": text,
            "usage": {"input_tokens": 4, "output_tokens": 3}}), "", code, False

    def agent_factory(options, **kwargs):
        agent = CpuAgent(options, process_runner=process, **kwargs)
        agents.append(agent)
        return agent

    monkeypatch.setattr("aka.agent.provider.Agent", agent_factory)

    def load_problem(provider, source, **kwargs):
        skeleton = CpuCampaign(directory)
        snapshot = kwargs.get("snapshot_directory") or directory / "snapshot"
        return LoadedProblem("synthetic_cpu", task_directory, task_directory / "reference.py",
            None, None, "none", agent=AgentProblem(()), initial_source=skeleton.problem.initial_source,
            evaluation=replace(skeleton.problem.evaluation, runtime_root=directory / "synthetic-evaluator"),
            snapshot_directory=snapshot)

    monkeypatch.setattr("aka.task.problem.legacy.LegacyProblemProvider.load", load_problem)
    # Public Task preparation belongs to the CPU fixture, while all outer
    # startup and Light construction still execute their actual implementation.
    monkeypatch.setattr("aka.workflow.legacy.startup.prepare_request",
                        lambda launch, problem, **kw: (launch.request, ()))
    import shutil
    original_which = shutil.which
    monkeypatch.setattr("shutil.which", lambda command, **kw: "/usr/bin/true" if command == "claude"
                        else original_which(command, **kw))

    def create(cls, request, problem, workflow, evaluation):
        campaign = CpuCampaign(directory)
        campaign.request = request
        campaign.problem = problem
        campaign.agent = workflow.agent
        campaign.candidate_workspace = workflow.candidate
        campaign.driver = workflow.driver
        campaign._module_resources = workflow.resources or campaign._module_resources
        campaign.max_iters = request.settings.max_iters
        campaign.token_budget = request.settings.token_budget
        campaign.handoff_resumes = 0
        campaign.sandbox_hardware = request.runtime.hardware
        campaign.module_options = replace(workflow.options, search=None)
        campaign.tool_runtime = SimpleNamespace(wiki=False, research_records=False, workflow=workflow)
        campaign.problem_provider.snapshot = lambda _: ProblemSnapshot(problem.snapshot_directory, "prepared-public-cpu")
        original_content = campaign.problem_provider.content
        def content(*args, **kwargs):
            value = original_content(*args, **kwargs)
            value.prepare_public = lambda: PublicInputContent(True, (
                WorkspaceFile("task.md", text="Synthetic CPU scalar fixture; no GPU qualification."),))
            return value
        campaign.problem_provider.content = content
        runtime = campaign._supervisor_runtime
        runtime.read_episode_journal = lambda _: {}
        runtime.read_experiment_records = lambda: ()
        runtime.read_episode_draft = lambda episode: campaign.candidate_workspace.read_source(
            campaign.workspace / ".atrex_long_horizon/drafts" / f"e{episode:04d}/workspace")
        runtime.close = lambda: None
        campaigns.append(campaign)
        return campaign

    monkeypatch.setattr("aka.workflow.light.campaign.ManagedCampaign.create", classmethod(create))
    return SimpleNamespace(directory=directory, campaigns=campaigns, agents=agents, calls=calls,
                           task=task_directory)


def native_selection(harness, *, paired=False):
    from aka.bootstrap.profile import resolve_profile
    from aka.bootstrap.workflow_cli import profile_path
    patches = [{"id": "resources", "config": {"plugins": [], "bundles": []}},
               {"id": "workflow", "config": {"startup": True, "search": {},
                "inherit_state": False, "output_limit_recovery_retries": 0,
                "episode_timeout_seconds": 60}}]
    if paired:
        patches.append({"id": "source", "config": {"kind": "kernel",
                        "kernel_demo": str(harness.directory / "input.py")}})
    selection = resolve_profile(profile_path("light"), patches=patches,
                                variables={"repo_root": str(harness.directory)})
    return replace(selection, resources=(), environment={**selection.environment,
        "ANTHROPIC_MODEL": "qwen3.8-max[1m]"})


def launch(harness, *, max_iters=0, provided_settings=None):
    from aka.contracts.workflow import WorkflowSettings, CampaignRuntimeSettings
    from aka.workflow.legacy.requests import create_launch
    return create_launch(harness.task, workspace_base=harness.directory / "runs",
        workspace_suffix="public_cpu", settings=WorkflowSettings(platform="cpu", framework="triton",
            max_iters=max_iters, token_budget=0, target_util=0, framework_baseline="never",
            convert_after=0, fast_episodes=0, handoff_resumes=0, setup_timeout=3600),
        runtime=CampaignRuntimeSettings(hardware="L20D", timeout=3600),
        provided_settings=provided_settings)


def schema_config(harness, *, paired=False, episodes=2):
    config = {"format": "aka.schema.launch.v1", "continuous": True,
              "limits": {"max_episodes": episodes}, "wiki_ids": [],
              "planner": {"timeout_seconds": 60, "max_repairs": 0}}
    if paired:
        evidence = harness.directory / "pairing.json"
        evidence.write_text(json.dumps({"purpose": "synthetic CPU entrypoint fixture only",
            "kernel_sha256": hashlib.sha256(BASELINE).hexdigest(), "device": "L20D"}))
        config["initial"] = {"schema": str(FIXTURE.resolve()),
            "source_files": {"kernel.py": hashlib.sha256(BASELINE).hexdigest()},
            "hardware": "L20D", "evidence": [str(evidence)]}
    return config
