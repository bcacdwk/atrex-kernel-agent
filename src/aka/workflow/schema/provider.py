"""The optional selection seam; the fixed business loop belongs to A5."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from importlib.machinery import PathFinder
from pathlib import Path
import re
import sys

from .release import SchemaUnavailable, verify_release

DEFAULT_BACKEND_FACTORY = "aka.workflow.schema.backend:create_workflow"

@dataclass(frozen=True)
class SchemaSelection:
    configuration: bytes
    origin: Path
    provenance: dict
    source_sha256: str

    @classmethod
    def load(cls, path, *, expected_sha256=None):
        provenance = verify_release()
        path = Path(path).expanduser().resolve(strict=True)
        raw = path.read_bytes()
        source_sha256 = "sha256:" + hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and source_sha256 != expected_sha256:
            raise SchemaUnavailable("Schema launch selection changed; continuation requires the original bytes")
        from kernelschema.contracts import loads
        value = loads(raw)
        if (not isinstance(value, dict) or set(value) != {"format", "request", "run_root", "backend"}
                or value["format"] != "aka.schema.selection.v1"):
            raise SchemaUnavailable("Schema config needs format, request, run_root and backend")
        for name in ("request", "run_root"):
            if not isinstance(value[name], str) or not value[name]:
                raise SchemaUnavailable(f"Schema {name} must be a path")
            value[name] = str((path.parent / value[name]).resolve())
        backend = value["backend"]
        if isinstance(backend, dict):
            backend.setdefault("factory", DEFAULT_BACKEND_FACTORY)
        if (not isinstance(backend, dict) or set(backend) != {"factory", "implementation_sha256", "parameters"}
                or not isinstance(backend["parameters"], dict)):
            raise SchemaUnavailable("Schema backend needs a pinned factory and parameters; A5 owns fixed execution")
        from kernelschema.contracts import jcs
        result = cls(jcs(value), path, provenance, source_sha256)
        result.request_document()  # Zero-fee Task/ref validation before startup.
        return result

    @property
    def document(self):
        from kernelschema.contracts import loads
        return loads(self.configuration)

    @property
    def identity(self):
        return "sha256:" + hashlib.sha256(self.configuration).hexdigest()

    def request_document(self):
        from kernelschema.artifacts import Bundle
        from kernelschema.contracts import loads, validate
        from kernelschema.materials import validate_request
        request_path = Path(self.document["request"])
        request = validate(loads(request_path.read_bytes()))
        dynamic = self.document["backend"]["parameters"].get("dynamic")
        initial_design = isinstance(dynamic, dict) and dynamic.get("start") == "design"
        if request.get("operation") != ("design" if initial_design else "implement"):
            raise SchemaUnavailable("AKA Schema selection requires implement, or an explicit dynamic design start")
        validate_request(Bundle(request_path.parent), request)
        return request

    def problem_provider(self, *, role="coder"):
        from .task import SchemaProblemProvider
        request = self.request_document()
        return SchemaProblemProvider(request["task"], request["scope"], request["materials"], role)

    def backend_factory(self):
        selected = self.document["backend"]
        target = selected["factory"]
        if not isinstance(target, str) or re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", target) is None:
            raise SchemaUnavailable("Schema backend factory must be an installed module:function")
        module_name, function = target.split(":")
        # find_spec('parent.child') imports parent. Resolve explicit import paths
        # without executing any candidate factory or its package initializers.
        search, spec = sys.path, None
        components = module_name.split(".")
        for index in range(len(components)):
            fullname = ".".join(components[:index + 1])
            loaded = sys.modules.get(fullname) if index < len(components) - 1 else None
            spec = loaded.__spec__ if loaded is not None else PathFinder.find_spec(fullname, search)
            if spec is None:
                break
            search = getattr(loaded, "__path__", spec.submodule_search_locations)
            if index < len(components) - 1 and search is None:
                spec = None
                break
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            raise SchemaUnavailable(f"Schema fixed backend is unavailable: {target}; A5 implementation is required")
        actual = "sha256:" + hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
        if actual != selected["implementation_sha256"]:
            raise SchemaUnavailable("Schema backend implementation differs from the fixed selection")
        factory = getattr(importlib.import_module(module_name), function, None)
        if not callable(factory):
            raise SchemaUnavailable("Schema backend factory is not callable")
        return factory

    def claude_profile(self):
        from kernelschema.artifacts import Bundle
        from kernelschema.runtime import ClaudeProfile
        request = self.request_document()
        bundle = Bundle(Path(self.document["request"]).parent)
        if request["operation"] == "design":
            fixed = self.document["backend"]["parameters"]["dynamic"].get("implementation")
            if not isinstance(fixed, dict) or "profile" not in fixed:
                raise SchemaUnavailable("dynamic design requires an explicit implementation profile")
            selected = fixed["profile"]
        else:
            selected = request["backend"]["profile"]
        profile = bundle.document(selected, "backend_profile")
        value = profile["parameters"].get("claude")
        if not isinstance(value, dict):
            raise SchemaUnavailable("Schema native profile needs fixed parameters.claude; no default Agent/model is substituted")
        selected = ClaudeProfile(**value)
        if selected.requested_model != profile["model"]:
            raise SchemaUnavailable("Schema profile and native model differ")
        return selected


@dataclass(frozen=True)
class NativeModules:
    """Borrowed original owners; no component or ledger is copied by A3."""
    agent: object
    source: object
    candidate: object
    executor: object
    execution_context: object
    evaluator_factory: object
    evaluation_defaults: object
    experiment_factory: object
    measurement_parallelism: object
    workflow_observer: object
    experiment_episode_evidence: object
    evaluation_episode_interpreter: object

    def light(self, *, agent, problem, executor, evaluator_factory, options=None):
        """Assemble original Light with explicitly replaced actual runtime services.

        Both Evaluator and CommandEvaluator come from evaluator_factory in the
        unchanged ManagedRuntimeFactory.open. A5 owns all charge/audit hooks.
        """
        from aka.workflow.light.provider import LightWorkflow, WorkflowOptions
        return LightWorkflow(agent, problem, self.source, self.candidate,
            executor=executor, execution_context=self.execution_context,
            experiment_factory=self.experiment_factory, evaluator_factory=evaluator_factory,
            evaluation_defaults=self.evaluation_defaults, measurement_parallelism=self.measurement_parallelism,
            workflow_observer=self.workflow_observer, episode_evidence_reader=self.experiment_episode_evidence,
            episode_evidence_interpreter=self.evaluation_episode_interpreter,
            options=options if options is not None else WorkflowOptions())


def create_workflow(selection, modules):
    """factory(selection, modules) -> original Workflow contract.

    A5 must replace evaluator AND command evaluator/executor in Light assembly,
    use one KS OwnerAccount, gate actual model/evaluation/audit operations and
    validate retained native locators. An A3 factory alone grants no qualification.
    """
    workflow = selection.backend_factory()(selection, modules)
    if any(not callable(getattr(workflow, name, None)) for name in ("campaign", "close")):
        raise SchemaUnavailable("Schema backend must return the original Workflow contract")
    return workflow
