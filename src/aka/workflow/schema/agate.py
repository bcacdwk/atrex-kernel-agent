"""Explicit real Agate service factory; consumes existing qualification only.

K6 validates the original full Task probe/baseline/four-negative receipts. A5
separately requires native AKA coder/reviewer permission/output captures. Neither
the K3 public profile nor a CPU test transport can sign this qualification.
"""
from pathlib import Path

from kernelschema.contracts import ProtocolError, UnsupportedError, digest, sha256, loads
from kernelschema.evaluation.channel import AgateChannel as KSChannel
from kernelschema.runtime import ClaudeProfile

from .evaluation import AgateChannel
from .session import NATIVE_PROFILE


def native_admission(context, reference):
    from .native_probe import implementation, references, validate_capture
    if reference is None:
        raise UnsupportedError("AKA native coder/reviewer qualification has not been selected")
    retained = [entry["file"] for entry in context.objects["materials"]["entries"] if not entry["roles"]]
    if reference not in retained:
        raise UnsupportedError("AKA native qualification must be retained as explicit host-only Materials (roles=[]) ")
    value = loads(context.bundle.read(reference))
    settings = context.objects["profile"]["parameters"]["claude"]
    expected = implementation()
    if (value.get("format") != "aka.native.qualification.v1" or value.get("profile") != NATIVE_PROFILE
            or value.get("claude_sha256") != digest(settings) or value.get("implementation") != expected
            or value.get("synthetic") is not False or set(value.get("roles", {})) != {"coder", "reviewer"}):
        raise UnsupportedError("original AKA qualification does not bind these native permissions/bytes/profile")
    profile = ClaudeProfile(**settings)
    if sha256(Path(profile.executable).read_bytes()) != profile.executable_sha256:
        raise ProtocolError("qualified native CLI executable changed")
    try:
        if any(ref not in retained for ref in references(value)):
            raise UnsupportedError("original native input/exec/invocation capture is missing from host-only Materials")
        for role, capture in value["roles"].items():
            validate_capture(context.bundle, capture, profile, role)
    except (KeyError, TypeError, ValueError) as error:
        raise UnsupportedError("AKA native qualification lacks valid original transport artifacts: " + str(error)) from error
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
        settings = self.service.channel.settings if self.service.channel is not None else None
        return {"factory": "aka.workflow.schema.agate", "fixed_evaluation_sha256": digest(settings)}

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
        self.channel = KSChannel(profile, Path(root) / "agate-jobs") if "evaluation" in profile["parameters"] else None
        self.native_qualification = parameters["native_qualification"]
        self.inner = None
        self.evaluation = EvaluationProxy(self)

    def identity(self):
        return {"evaluation": self.evaluation.identity(), "native_qualification": self.native_qualification}

    def admission(self, context):
        try:
            native_admission(context, self.native_qualification)
            if self.channel is None:
                raise UnsupportedError("select the original K6 full-Task Agate channel and six-job qualification")
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
