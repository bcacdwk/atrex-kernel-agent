"""Schema context for the normal Light workflow, with no alternate evaluator.

This module deliberately has no KernelSchema import at module load time. Raw
Schema files are useful on their own; writing and legacy strict runs are separate
optional operations.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path


def _digest(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def schema_context(schema, docs=()):
    """Read the selected raw design and explicitly requested supporting files.

    A directory contributes YAML nodes and its dispatch.md. Documentation is
    opt-in, so selecting a design never scans the surrounding project or adds
    the entire Schema language manual to each coding session.
    """
    path = Path(schema).expanduser().resolve(strict=True)
    if path.is_dir():
        paths = sorted(p for p in path.rglob("*") if p.is_file()
                       and (p.suffix.lower() in {".yaml", ".yml"} or p.name == "dispatch.md"))
        files = [{"name": p.relative_to(path).as_posix(), "text": p.read_text(encoding="utf-8")}
                 for p in paths]
    else:
        files = [{"name": path.name, "text": path.read_text(encoding="utf-8")}]
    if not files or not any(item["text"].strip() for item in files):
        raise ValueError("Schema input must contain a design file or YAML nodes")
    for index, document in enumerate(docs):
        document = Path(document).expanduser().resolve(strict=True)
        files.append({"name": f"docs/{index + 1}-{document.name}",
                      "text": document.read_text(encoding="utf-8")})
    return {"files": files, "identity": _digest(files)}


def context_prompt(context):
    if context.get("identity") != _digest(context["files"]):
        raise ValueError("Schema context identity changed")
    parts = ["Selected Schema design (" + context["identity"] + ").\n"
        "Implement this design for the operator contract in the task below. "
        "Keep the same design across coding episodes; refine its implementation "
        "using the normal AKA tools and measured correctness/performance feedback. "
        "Honor the task's interface, shapes, numerical requirements and framework. "
        "If the design conflicts with that contract, report the conflict. "
        "Schema text is design input, not evaluation evidence. Submit code and "
        "the usual AKA episode report; describe measured failures and remaining "
        "limitations truthfully.\n"]
    for item in context["files"]:
        parts.append("\n--- Schema input: " + item["name"] + " ---\n" + item["text"])
    return "\n".join(parts) + "\n\n--- AKA task and episode ---\n"


class _Runtime:
    def __init__(self, native, prefix, budget=None, cancel=None):
        self.native, self.prefix, self.budget, self.cancel = native, prefix, budget, cancel

    def __getattr__(self, name):
        return getattr(self.native, name)

    def run(self, request):
        request = replace(request, prompt=self.prefix + request.prompt)
        if self.budget is None:
            return self.native.run(request)
        return self.budget.model(lambda reservation: self.native.run(request), cancel=self.cancel)


class _Session:
    def __init__(self, native, prefix, budget=None):
        self.native, self.prefix, self.budget = native, prefix, budget

    def __getattr__(self, name):
        return getattr(self.native, name)

    def invoke(self, prompt, **kwargs):
        # A resumed session already contains the design; a fresh episode gets it
        # again. Avoid paying to resend the same document in every repair turn.
        prompt = prompt if kwargs.get("resume") else self.prefix + prompt
        if self.budget is None:
            return self.native.invoke(prompt, **kwargs)
        return self.budget.model(lambda reservation: self.native.invoke(prompt, **kwargs), cancel=self.native.close)


class SchemaAgent:
    """Borrow the original Agent, including its tools, settings and receipts."""
    def __init__(self, native, context, budget=None):
        self.native, self.context, self.budget = native, context, budget
        self.prefix = context_prompt(context)

    def __getattr__(self, name):
        return getattr(self.native, name)

    @property
    def configuration_identity(self):
        return _digest([self.native.configuration_identity, self.context["identity"]])

    def runtime(self, runtime_id):
        return _Runtime(self.native.runtime(runtime_id), self.prefix, self.budget, self.native.close)

    def open_session(self, *args, **kwargs):
        return _Session(self.native.open_session(*args, **kwargs), self.prefix, self.budget)


def select(profile, *, schema, docs=(), patch_files=(), variables=None, owner_account=None,
           owner_limits=None, owner_run_id=None, workspace=None):
    """Resolve a normal composition and replace only its workflow decorator."""
    from aka.bootstrap.profile import resolve_profile
    launch = resolve_profile(profile, patch_files=patch_files, variables=variables or {})
    row = launch.composition.row("workflow")
    if row is None or row.disabled or row.name != "aka.workflow.light.plugin":
        raise ValueError("--schema requires the native Light workflow composition")
    context = schema_context(schema, docs)
    config = {**row.config, "schema_context": context}
    if owner_account is not None:
        config["owner_account"] = str(Path(owner_account).expanduser().resolve(strict=True))
        if workspace is None:
            raise ValueError("parent accounting needs the native campaign workspace")
        config.update(owner_limits=owner_limits, owner_workspace=str(Path(workspace).resolve()))
        if owner_run_id is not None:
            config["owner_run_id"] = owner_run_id
    replacement = replace(row, name="aka.workflow.schema.standard_plugin", config=config)
    composition = replace(launch.composition, rows=tuple(
        replacement if item.id == row.id else item for item in launch.composition.rows))
    return replace(launch, composition=composition)


def run(profile, invocation, *, schema, docs=(), patch_files=(), variables=None, owner_account=None,
        owner_limits=None, owner_run_id=None):
    from aka.bootstrap.host import run_selection
    from aka.contracts.workflow_launch import decode_launch
    selection = select(profile, schema=schema, docs=docs, patch_files=patch_files,
                       variables=variables, owner_account=owner_account, owner_limits=owner_limits,
                       owner_run_id=owner_run_id, workspace=decode_launch(invocation.argv).request.workspace)
    return run_selection(selection, invocation)


def collect_result(workspace, *, exit_code, schema_identity, stop_reason=None):
    """Project retained native facts without inferring correctness from exit 0."""
    workspace = Path(workspace).resolve()
    versions = []
    for directory in (workspace / ".atrex_versions", workspace / "memory"):
        if directory.is_dir():
            for path in sorted(directory.glob("v[0-9]*.json")):
                value = json.loads(path.read_text(encoding="utf-8"))
                versions.append({"path": str(path), "record": value})
    state_path = workspace / ".atrex_long_horizon/state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else None
    artifacts = {"workspace": str(workspace), "versions": [v["path"] for v in versions]}
    for name, relative in (("state", ".atrex_long_horizon/state.json"),
                           ("solution", "solution.json"), ("kernel", "kernel.py"),
                           ("source_policy", ".atrex_source.json")):
        if (workspace / relative).is_file():
            artifacts[name] = str(workspace / relative)
    native_result = workspace / "schema-native-result.json"
    if native_result.is_file():
        outcome = json.loads(native_result.read_text())
        if outcome.get("schema_identity") == schema_identity:
            stop_reason = outcome.get("stop_reason", stop_reason)
        artifacts["native_result"] = str(native_result)
        if outcome.get("budget_receipts"):
            artifacts["budget_receipts"] = outcome["budget_receipts"]
    feedback_files = [v["path"] for v in versions]
    for name in ("attempt.json", "public-candidate.json"):
        feedback_files.extend(str(path) for path in sorted(
            (workspace / ".atrex_long_horizon/episodes").glob("*/" + name)))
    if native_result.is_file():
        feedback_files.append(str(native_result))
    return {"status": "completed" if exit_code == 0 else "failed", "exit_code": exit_code,
        "stop_reason": stop_reason or ("native workflow completed" if exit_code == 0 else "native workflow failed"),
        "workspace": str(workspace), "schema_identity": schema_identity,
        "evaluation_scope": "AKA native Atrex-Bench correctness/performance for the supplied reference, input and shapes; native ABBA promotion",
        "versions": versions, "native_state": state, "artifacts": artifacts,
        "feedback_files": feedback_files}


def implement(*, schema, problem, repo_root, workspace, platform, framework="Triton",
              agent_cli="claude", max_iters=4, docs=(), profile="light", patches=(),
              settings=None, runtime=None, prepare_only=False, owner_account=None,
              owner_limits=None, owner_run_id=None, _launch=None):
    """Synchronous KS -> AKA entry. Workspace is the exact campaign directory.

    ``prepare_only`` loads the original Task and composition without booting
    services, invoking a model, or submitting a GPU job. Native settings and
    runtime options are the same public DTO fields as ``aka optimize``.
    """
    from aka.bootstrap.workflow_cli import profile_path
    from aka.contracts.startup import Invocation
    from aka.contracts.workflow import CampaignRequest, CampaignRuntimeSettings, ProblemSource, WorkflowSettings
    from aka.contracts.workflow_launch import WorkflowLaunch, encode_launch
    from aka.task.problem.legacy import LegacyProblemProvider
    root = Path(repo_root).expanduser().resolve(strict=True)
    workspace = Path(workspace).expanduser().resolve()
    problem = Path(problem).expanduser().resolve(strict=True)
    values = {"platform": platform, "framework": framework, "agent_cli": agent_cli,
              "max_iters": max_iters, "framework_baseline": "never", "convert_after": 0,
              "target_util": 0, **(settings or {})}
    workflow_settings = WorkflowSettings(**values)
    request = CampaignRequest(ProblemSource(problem), workspace, workspace.name,
                              workflow_settings, CampaignRuntimeSettings(**(runtime or {})))
    launch = WorkflowLaunch(request, workspace.parent, exact_workspace=True)
    if _launch is not None:
        # The automatic CLI entry retains native provided-field/default semantics
        # and its original ProblemSource. Only this version's workspace changes.
        request = replace(_launch.request, workspace=workspace, name=workspace.name)
        launch = replace(_launch, request=request, exact_workspace=True, restart_arguments=())
        workflow_settings = request.settings
    invocation = Invocation(encode_launch(launch))
    selection = select(profile_path(profile), schema=schema, docs=docs, patch_files=patches,
                       variables={"repo_root": str(root)}, owner_account=owner_account,
                       owner_limits=owner_limits, owner_run_id=owner_run_id, workspace=workspace)
    context = selection.composition.row("workflow").config["schema_context"]
    # Original task parser: no K1 acceptance suite or synthetic qualification.
    try:
        loaded = LegacyProblemProvider().load(problem,
            optimization_mode=workflow_settings.optimization_mode,
            expose_evaluator_shapes=workflow_settings.expose_evaluator_shapes)
    except SystemExit as error:
        raise ValueError(str(error)) from error
    if loaded.evaluation.runtime_root is None:
        raise ValueError("Schema standard implementation requires a native Atrex-Bench operator directory")
    if prepare_only:
        return {"status": "prepared", "stop_reason": "prepare_only", "workspace": str(workspace),
                "schema_identity": context["identity"], "problem": str(problem),
                "evaluation_scope": "AKA native Atrex-Bench", "versions": [], "artifacts": {},
                "exit_code": None, "invocation": list(invocation.argv)}
    request_identity = _digest([list(invocation.argv), [
        [row.id, row.name, row.config, row.disabled] for row in selection.composition.rows]])
    destination = workspace / "schema-result.json"
    if destination.is_file():
        previous = json.loads(destination.read_text())
        if previous.get("request_identity") != request_identity:
            raise ValueError("existing Schema implementation has a different request; choose a new workspace")
        return previous
    from aka.bootstrap.host import run_selection
    failure = None
    try:
        exit_code = run_selection(selection, invocation)
    except Exception as error:
        # Preserve the native records so a planner can consume an unsuccessful
        # implementation too. Signals/KeyboardInterrupt retain native recovery.
        exit_code, failure = 1, {"type": type(error).__name__, "message": str(error)}
    result = collect_result(workspace, exit_code=exit_code, schema_identity=context["identity"])
    from aka.execution.recovery import ENVIRONMENT_TEMPFAIL
    if exit_code == ENVIRONMENT_TEMPFAIL:
        result["status"] = "blocked"
        result["stop_reason"] = "native environment unavailable; settle the original workspace"
    elif exit_code in {130, 143}:
        result["status"] = "interrupted"
    if failure is not None:
        result["error"] = failure
        result["stop_reason"] = failure["type"]
        if failure["type"] in {"BudgetDenied", "ExecutionPending", "EnvironmentUnavailable"}:
            result["status"] = "blocked"
    config = selection.composition.row("workflow").config
    if config.get("owner_account"):
        from .standard_budget import StandardBudget
        budget = StandardBudget(config["owner_account"], workspace=workspace,
            identity=context["identity"], limits=config.get("owner_limits"),
            run_id=config.get("owner_run_id"))
        result["usage"] = {"scope": "native_stage", "unsettled": budget.unsettled()}
    result["request_identity"] = request_identity
    if workspace.is_dir():
        result["artifacts"]["feedback"] = str(destination)
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
