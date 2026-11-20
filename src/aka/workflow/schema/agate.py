"""Explicit real Agate service factory; consumes existing qualification only.

K6 validates the original full Task probe/baseline/four-negative receipts. A5
separately requires native AKA coder/reviewer permission/output captures. Neither
the K3 public profile nor a CPU test transport can sign this qualification.
"""
from pathlib import Path
from types import SimpleNamespace

from kernelschema.contracts import ProtocolError, UnsupportedError, digest, sha256, loads
from kernelschema.evaluation.channel import AgateChannel as KSChannel
from kernelschema.runtime import ClaudeProfile

from .evaluation import AgateChannel
from .session import RoleSession, NATIVE_PROFILE
from .state import invocation_from_document


def native_admission(context, reference):
    from . import session, stdin_exec
    from aka.agent import native
    if reference is None:
        raise UnsupportedError("AKA native coder/reviewer qualification has not been selected")
    retained = [entry["file"] for entry in context.objects["materials"]["entries"] if not entry["roles"]]
    if reference not in retained:
        raise UnsupportedError("AKA native qualification must be retained as explicit host-only Materials (roles=[]) ")
    value = loads(context.bundle.read(reference))
    settings = context.objects["profile"]["parameters"]["claude"]
    expected = {m.__name__: sha256(Path(m.__file__).read_bytes()) for m in (session, stdin_exec, native)}
    if (value.get("format") != "aka.native.qualification.v1" or value.get("profile") != NATIVE_PROFILE
            or value.get("claude_sha256") != digest(settings) or value.get("implementation") != expected
            or value.get("synthetic") is not False or set(value.get("roles", {})) != {"coder", "reviewer"}):
        raise UnsupportedError("original AKA qualification does not bind these native permissions/bytes/profile")
    profile = ClaudeProfile(**settings)
    if sha256(Path(profile.executable).read_bytes()) != profile.executable_sha256:
        raise ProtocolError("qualified native CLI executable changed")
    for role, capture in value["roles"].items():
        if capture["invocation"] not in retained:
            raise UnsupportedError("original native invocation capture is missing from host-only Materials")
        document = loads(context.bundle.read(capture["invocation"]))
        invocation = invocation_from_document(document)
        locator = capture["locator"]
        if locator["role"] != role or locator["profile"] != NATIVE_PROFILE or locator["native_context_resume"] is not False:
            raise ProtocolError("native qualification role/locator mismatch")
        inspector = object.__new__(RoleSession)
        inspector.role = SimpleNamespace(document={"role": role}, assignment_sha256=locator["assignment_sha256"])
        inspector.profile, inspector.workspace = profile, Path(locator["workspace"])
        inspector._session = SimpleNamespace(session_id=locator["session_id"])
        errors, output = inspector._inspect(invocation)
        if errors or output is None or invocation.terminal_usage.total_tokens is None:
            raise UnsupportedError("AKA native original permission/output/usage capture is incomplete: " + ", ".join(errors))
    return {"status": "pass", "reason": "original AKA native role/profile captures verified"}


class EvaluationProxy:
    def __init__(self, service):
        self.service = service

    @property
    def executor(self):
        if self.service.inner is None:
            raise UnsupportedError("original Agate qualification must be admitted before evaluation")
        return self.service.inner.executor

    def identity(self):
        return {"factory": "aka.workflow.schema.agate", "fixed_evaluation_sha256": digest(self.service.channel.settings)}

    def prepare(self, staged, operation):
        return self.service.inner.prepare(staged, operation)

    def submit(self, staged, operation, *, authorize=None):
        return self.service.inner.submit(staged, operation, authorize=authorize)

    def worker(self, facts):
        return self.service.inner.worker(facts)


class Services:
    comparison = None  # A separate same-allocation protocol is required for ABBA.

    def __init__(self, profile, root, parameters):
        if set(parameters) != {"native_qualification"}:
            raise ProtocolError("AKA Agate services need exactly native_qualification Ref or null")
        self.channel = KSChannel(profile, Path(root) / "agate-jobs")
        self.native_qualification = parameters["native_qualification"]
        self.inner = None
        self.evaluation = EvaluationProxy(self)

    def identity(self):
        return {"evaluation": self.evaluation.identity(), "native_qualification": self.native_qualification}

    def admission(self, context):
        try:
            native_admission(context, self.native_qualification)
            result = self.channel.admission(context)
            if result["status"] != "pass":
                return result
            environment, frozen = self.channel.capture
            self.inner = AgateChannel(self.channel.executor, profile=self.channel.settings["remote"],
                                      environment=environment, frozen_inputs=frozen)
            return result
        except UnsupportedError as error:
            return {"status": "unknown", "reason": str(error)}


def workflow_services(profile, root, parameters):
    return Services(profile, root, parameters)
