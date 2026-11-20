"""Installed AKA fixed backend. Both CLI directions use FixedController.

No planner, environment discovery, model/GPU probe or second budget owner is
created here. An explicitly pinned service factory admits actual transports.
"""
from __future__ import annotations

import importlib
from importlib.machinery import PathFinder
from pathlib import Path
import sys

from kernelschema.api import open_run, prepare_run
from kernelschema.artifacts import Bundle
from kernelschema.contracts import ProtocolError, digest, sha256
from kernelschema.evaluation import LocalExecutor
from kernelschema.runtime import ClaudeProfile

from .controller import FixedController
from .evaluation import LocalChannel, SchemaExecution, SchemaEvaluatorFactory
from .provider import NativeModules
from .session import native_agent, native_capabilities
from .task import SchemaProblemProvider


class LocalServices:
    """Concrete CPU diagnostics; they grant no CUDA/native-model qualification."""
    comparison = None

    def __init__(self, root):
        self.evaluation = LocalChannel(LocalExecutor(Path(root) / "jobs"))

    def identity(self):
        return {"kind": "CPU diagnostics only", "evaluation": self.evaluation.identity()}

    def admission(self, context):
        return {"status": "unknown", "reason": "no qualified AKA native/model and CUDA full-Task service profile selected"}


def _factory(selection):
    if not isinstance(selection, dict) or set(selection) != {"factory", "implementation_sha256", "parameters"}:
        raise ProtocolError("AKA services need installed factory, implementation_sha256 and parameters")
    module, separator, name = selection["factory"].partition(":")
    if not separator or not name.isidentifier() or not all(p.isidentifier() for p in module.split(".")):
        raise ProtocolError("AKA service factory must be installed module:function")
    search, spec = sys.path, None
    for index, part in enumerate(module.split(".")):
        full = ".".join(module.split(".")[:index + 1])
        loaded = sys.modules.get(full) if index < len(module.split(".")) - 1 else None
        spec = loaded.__spec__ if loaded else PathFinder.find_spec(full, search)
        if spec is None:
            raise ProtocolError("selected AKA service is not installed")
        search = getattr(loaded, "__path__", spec.submodule_search_locations)
    if not spec.origin or sha256(Path(spec.origin).read_bytes()) != selection["implementation_sha256"]:
        raise ProtocolError("installed AKA service differs from the frozen profile")
    return getattr(importlib.import_module(module), name)


def services(profile, root):
    selected = profile["parameters"].get("aka", {}).get("services")
    value = LocalServices(root) if selected is None else _factory(selected)(profile, Path(root), selected["parameters"])
    for name in ("identity", "admission"):
        if not callable(getattr(value, name, None)):
            raise ProtocolError("AKA services lack " + name)
    if not hasattr(value, "evaluation"):
        raise ProtocolError("AKA services lack the original A2 evaluation channel")
    return value


def implementation(profile):
    from . import state, calls, comparison, controller, projection, evaluation, session, preparation
    from aka.workflow.light.long_horizon import campaign, recorded_verifier
    from aka.agent import native
    from .release import verify_release
    selected = profile["parameters"].get("aka", {}).get("services")
    if selected is not None:
        _factory(selected)
    return {"backend": "aka", "version": "aka.fixed.v1", "profile": digest(profile),
        "ks_manifest": verify_release()["manifest_sha256"],
        "schema_source": {p.name: sha256(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "implementation": {module.__name__: sha256(Path(module.__file__).read_bytes()) for module in
            (sys.modules[__name__], state, calls, comparison, controller, projection, evaluation, session,
             preparation, campaign, recorded_verifier, native)}}


class FixedWorkflow:
    """Retain the original Light Workflow and ManagedCampaign objects."""
    def __init__(self, selection, modules, *, context=None, supplied_services=None, continuing=None, handoff=None):
        self.selection, self.modules, self.context = selection, modules, context
        self.handoff = handoff
        request = context.request if context else selection.request_document()
        root = context.root if context else Path(selection.document["run_root"])
        profile = context.objects["profile"] if context else Bundle(Path(selection.document["request"]).parent).document(request["backend"]["profile"])
        if context is None:
            identity = implementation(profile)
            continuing = root.exists()
            self.context = open_run("implement", root, backend_identity=identity) if continuing else prepare_run(
                "implement", request=selection.document["request"], out=root, backend_identity=identity)
        elif continuing is None:
            continuing = (context.root / ".runtime/state/aka-fixed-backend.json").exists()
        self.continuing = continuing
        self.services = supplied_services or services(profile, root / ".runtime/aka-services")
        self.execution = SchemaExecution(root / ".runtime/aka-evaluations", self.services.evaluation)
        self.controller = None
        def operation(request):
            if self.controller is None:
                raise ProtocolError("native evaluator used before its original fixed owner")
            return self.controller.resolve_evaluation(request)
        self.factory = SchemaEvaluatorFactory(execution=self.execution, operation=operation)
        from aka.workflow.light.provider import WorkflowOptions
        self.native = modules.light(agent=modules.agent,
            problem=SchemaProblemProvider(request["task"], request["scope"], request["materials"]),
            executor=self.execution, evaluator_factory=self.factory,
            options=WorkflowOptions(file_system="single_file", output_limit_recovery_retries=0))

    def _load_problem(self, request):
        return self.native._load_problem(request)

    def _campaign(self, request, problem):
        if self.controller is not None:
            raise ProtocolError("fixed AKA permits one serial native campaign per workflow")
        campaign = self.native._campaign(request, problem)
        self.controller = FixedController(self.context, campaign, self.execution, self.services,
            continuing=self.continuing, selection=self.selection.identity if self.selection else None, handoff=self.handoff)
        campaign.run = self.controller.execute
        campaign.schema_controller = self.controller
        return campaign

    def campaign(self, request):
        return self._campaign(request, self._load_problem(request))

    def close(self):
        self.native.close()


def create_workflow(selection, modules):
    """A3's installed default factory, returning the retained Workflow contract."""
    parameters = selection.document["backend"]["parameters"]
    if set(parameters) == {"dynamic"}:
        from .dynamic import DynamicWorkflow
        return DynamicWorkflow(selection, modules)
    if parameters:
        raise ProtocolError("fixed AKA execution parameters belong to the immutable backend profile parameters.aka")
    return FixedWorkflow(selection, modules)


def native_modules(profile):
    """Original native default owners for the synchronous KS -> AKA direction."""
    from aka.contracts.evaluation import EvaluationConfiguration
    from aka.contracts.execution import ExecutionContext
    from aka.task.source.legacy import LegacySourceProvider
    from aka.task.candidate_workspace.provider import GitCandidateWorkspace
    from aka.experiment.factory import LegacyExperimentFactory
    from aka.experiment.episode_evidence import LocalExperimentEpisodeEvidenceReader
    from aka.evaluation.episode_evidence import LegacyEvaluationEpisodeInterpreter
    return NativeModules(native_agent(ClaudeProfile(**profile["parameters"]["claude"])), LegacySourceProvider(),
        GitCandidateWorkspace(), None, ExecutionContext(), None, EvaluationConfiguration(),
        LegacyExperimentFactory(), 1, None, LocalExperimentEpisodeEvidenceReader(), LegacyEvaluationEpisodeInterpreter())


def native_request(context):
    from aka.contracts.workflow import CampaignRequest, ProblemSource, WorkflowSettings, CampaignRuntimeSettings
    configuration = context.objects["profile"]["parameters"].get("aka", {}).get("native", {})
    if not isinstance(configuration, dict) or set(configuration) - {"workspace", "name", "settings", "runtime"}:
        raise ProtocolError("AKA native configuration has unknown fields")
    workspace = Path(configuration.get("workspace", "native"))
    workspace = (workspace if workspace.is_absolute() else context.root / workspace).resolve()
    settings = {"platform": "H20", "framework": "triton", "framework_baseline": "never", "convert_after": 0,
                "max_iters": 20, "target_util": 0, **configuration.get("settings", {})}
    for name in ("framework_baseline_reviewers", "fast_episode_reviewers", "full_episode_reviewers"):
        if name in settings:
            settings[name] = tuple(settings[name])
    runtime = dict(configuration.get("runtime", {}))
    if isinstance(runtime.get("min_improvement_pct"), str):
        runtime["min_improvement_pct"] = float(runtime["min_improvement_pct"])
    if isinstance(settings.get("target_util"), str):
        settings["target_util"] = float(settings["target_util"])
    return CampaignRequest(ProblemSource(context.root, context.root / ".runtime/aka-problem"), workspace,
        configuration.get("name", "schema_fixed"), WorkflowSettings(**settings),
        CampaignRuntimeSettings(**runtime))


class Backend:
    def capabilities(self, profile):
        return native_capabilities()

    def identity(self, profile):
        return implementation(profile)

    def validate_resume(self, context):
        from kernelschema.budget import OwnerAccount
        account = OwnerAccount.for_context(context, create=False)
        path = context.root / ".runtime/state/aka-fixed-backend.json"
        if not path.exists() and (context.intents() or account.operations(context.run_id)):
            raise ProtocolError("original AKA state is missing; settlement cannot reset the owner")

    def execute(self, context, *, continuing=False):
        # KS already owns the RunContext and outer serial request. The same
        # authority is attached by FixedController; no nested api.implement call.
        from aka.workflow.legacy.public_state import campaign_lease
        modules = native_modules(context.objects["profile"])
        workflow = FixedWorkflow(None, modules, context=context, continuing=continuing)
        request = native_request(context)
        try:
            with campaign_lease(request.workspace, writer=True):
                campaign = workflow.campaign(request)
                campaign.prepare()
                return campaign.run()
        finally:
            workflow.close()
            modules.agent.close()
            modules.experiment_factory.close()


def workflow_backend():
    """KS's explicit installed implement/aka factory; no second outer loop."""
    return Backend()
