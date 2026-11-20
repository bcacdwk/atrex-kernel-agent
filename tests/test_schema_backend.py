"""A5 CPU engineering: real native owners, explicitly synthetic GPU/model service.

No test changes A2 CPU projection. FakeCudaExecutor manufactures transport facts
solely to make the qualified native branches reachable without CUDA or a model.
"""
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from kernelschema import fidelity
from kernelschema.api import prepare_run, open_run
from kernelschema.artifacts import Bundle, Publication, durable_write
from kernelschema.contracts import PROTOCOL, digest, jcs, loads, sha256
from kernelschema.evaluation.task import verify_stage
from aka.workflow.schema.backend import FixedWorkflow, native_modules, native_request
from aka.workflow.schema.session import native_agent
from aka.workflow.schema.evaluation import LocalChannel

from test_schema_optional_module import fixed

CODE = "# synthetic complete candidate; never executed or GPU-qualified\nclass Model:\n    def forward(self, x, out):\n        return out\n"


def report(publication, *, operation, staged, latency=10.0):
    manifest = staged["manifest"]
    root = Path(staged["root"])
    spec = loads((root / "inputs.json").read_bytes())
    cases = spec["cases"]
    env = {"mode": "cuda", "fake_transport": "A5 CPU test, no real device"}
    timing = {"complete": True, "summaries": [{"case_id": c["case_id"], "models": {"A": {"median_ns": int(latency * 1000)}}} for c in cases]}
    worker = {"environment": env, "source": manifest["source"], "evaluator": manifest["evaluator"],
        "loaded": manifest["files"], "frozen_inputs_sha256": sha256(b"explicit fake frozen inputs"),
        "result": {"build": "pass", "correctness": {"passed": True, "basic_calls": [{"case_id": c["case_id"]} for c in cases]}, "timing": timing}}
    raw = publication.put_bytes("synthetic-worker.json", jcs(worker))
    frozen = publication.put_bytes("synthetic-inputs.json", b"explicit fake frozen inputs")
    environment = publication.put_bytes("synthetic-environment.json", jcs(env))
    samples = publication.put_bytes("synthetic-timing.json", jcs(timing))
    check = {"status": "pass", "receipt": raw, "reason": "explicit test fake transport, no real qualification"}
    value = {"protocol": PROTOCOL, "kind": "evaluation", "operation_id": operation,
        "subject": {"task_sha256": manifest["task"]["sha256"], "source_sha256": manifest["source"]["sha256"],
            "evaluator_sha256": manifest["evaluator"]["sha256"], "inputs_sha256": manifest["inputs"]["sha256"],
            "timing_sha256": manifest["timing"]["sha256"], "environment_sha256": environment["sha256"]},
        "build": check, "correctness": check, "performance": check,
        "case_ids": sorted(c["case_id"] for c in cases), "samples": samples, "environment": environment,
        "raw_receipts": [raw, frozen], "usage_receipts": [],
        "extensions": {"information": {"fake_transport": "A5 CPU only", "performance_goal": "unmet"}}}
    ref = publication.put_json("evaluation.json", value)
    publication.commit([ref], subjects=[manifest["task"], manifest["source"]])
    return ref, worker, raw


class FakeCudaExecutor:
    def __init__(self, root):
        self.root = Path(root)
        self.submissions = []

    def identity(self):
        return {"name": "explicit synthetic A5 CUDA transport, no real GPU"}

    def locator(self, op):
        return {"root": str(self.root / op), "synthetic": True}

    def prepare(self, staged, **kwargs):
        verify_stage(staged)

    def submit(self, staged, **operation):
        path = self.root / operation["operation_id"]
        if not (path / "job.json").exists():
            durable_write(path / "job.json", jcs({"stage": staged, "operation": operation}))
            self.submissions.append(operation["operation_id"])
        return self.query(operation["operation_id"])

    def query(self, operation):
        job = loads((self.root / operation / "job.json").read_bytes())
        # Actual SchemaExecution retains the real source/evaluator stage. This
        # test transport alone supplies synthetic CUDA observations.
        manifest = job["stage"]["manifest"]
        cases = loads((Path(job["stage"]["root"]) / "inputs.json").read_bytes())["cases"]
        worker = {"environment": {"mode": "cuda", "fake_transport": "A5 CPU test, no real device"},
            "source": manifest["source"], "evaluator": manifest["evaluator"], "loaded": manifest["files"],
            "frozen_inputs_sha256": sha256(b"explicit fake frozen inputs"),
            "result": {"build": "pass", "correctness": {"passed": True, "basic_calls": [{"case_id": c["case_id"]} for c in cases]},
                "timing": {"complete": True, "summaries": [{"case_id": c["case_id"], "models": {"A": {"median_ns": 10000}}} for c in cases]}}}
        return {"state": "terminal", "receipt_id": operation + "-receipt", "worker": worker}

    def settle(self, operation, publication):
        root = self.root / operation
        if (root / "absorption.json").exists():
            return loads((root / "absorption.json").read_bytes())
        job = loads((root / "job.json").read_bytes())
        ref, worker, raw = report(publication, operation=operation, staged=job["stage"])
        row = {"producer": "A5-explicit-fake-cuda", "operation_id": operation, "role": "evaluator", "purpose": "synthetic full Task",
            "receipt_id": operation + "-receipt", "sequence": 1, "receipt": raw, "state": "settled",
            "calls": 0, "evaluator_jobs": 1, "tokens": 0, "gpu_ms": 0}
        terminal = {"state": "terminal", "report": ref, "usage_row": row}
        durable_write(root / "absorption.json", jcs(terminal))
        return terminal


class FakeServices:
    comparison = None

    def __init__(self, root):
        self.evaluation = LocalChannel(FakeCudaExecutor(root))

    def identity(self):
        return self.evaluation.identity()

    def admission(self, run):
        return {"status": "pass", "reason": "explicit synthetic CPU fixture, not deployed qualification"}


class FakeComparison:
    """Explicit one-allocation fake, with each full original Task/source stage."""
    def __init__(self, root, *, candidate_latency=5):
        self.root, self.latency, self.submissions = Path(root), candidate_latency, []

    def admission(self, context):
        return {"status": "pass", "reason": "synthetic allocation only"}

    def locator(self, operation):
        return {"root": str(self.root / operation), "synthetic_allocation": True}

    def prepare(self, context, spec):
        from aka.workflow.schema.state import put_once
        put_once(self.root / spec["operation_id"] / "spec.json", spec)

    def validate(self, context, spec):
        assert loads((self.root / spec["operation_id"] / "spec.json").read_bytes()) == spec

    def submit(self, context, spec, *, authorize):
        assert authorize()
        self.submissions.append(spec["operation_id"])

    def settle(self, context, spec, publication):
        path = self.root / spec["operation_id"] / "terminal.json"
        if path.exists():
            return loads(path.read_bytes())
        from kernelschema.evaluation import stage
        rows = []
        allocation = "explicit-synthetic-allocation-" + spec["operation_id"]
        for index, entry in enumerate(spec["schedule"]):
            source = spec[entry["revision"]]
            staged = stage(context.bundle, spec["task"], source, self.root / spec["operation_id"] / str(index), scope=spec["scope"])
            p = Publication(context.root, digest(spec["operation_id"] + str(index))[7:39])
            reference, _, _ = report(p, operation=spec["operation_id"], staged=staged,
                                     latency=self.latency if entry["revision"] == "candidate" else 10)
            rows.append({**entry, "allocation_id": allocation, "report": reference})
        receipt = {"operation_id": spec["operation_id"], "spec_sha256": digest(spec),
            "allocation_id": allocation, "same_allocation": True, "measurement_scope": "full-task",
            "schedule": spec["schedule"], "runs": rows}
        raw = publication.put_bytes("synthetic-comparison.json", jcs(receipt))
        publication.commit([raw])
        row = {"producer": "A5-explicit-fake-comparison", "operation_id": spec["operation_id"], "role": "evaluator",
            "purpose": "synthetic comparison", "receipt_id": raw["sha256"], "sequence": 1, "receipt": raw, "state": "settled",
            "calls": 0, "evaluator_jobs": 1, "tokens": 0, "gpu_ms": 0}
        terminal = {"state": "terminal", "usage_row": row, "comparison": receipt, "evidence": [raw]}
        durable_write(path, jcs(terminal))
        return terminal

    def measurement(self, context, spec, row):
        value = self.latency if row["revision"] == "candidate" else 10
        return {"measurement_scope": "full-task", "latency_us_geomean": value,
            "latency_us_by_shape": {cid: value for cid in context.objects["scope"]["coverage_ids"]}}


def model(run, trace, *, reviews=("pass",), candidates=(CODE,), unknown_tokens=False):
    def invoke(command, *, cwd, timeout, env):
        payload = loads(Path(command[3]).read_bytes())
        role = payload["role"]
        argv = command[command.index("--") + 1:]
        session = argv[argv.index("--session-id") + 1]
        intent = next(i for i in run.intents() if i["submission_key"] == session)
        count = sum(item["role"] == role for item in trace)
        trace.append({"role": role, "source": intent["source"], "session": session, "input": payload,
                      "authority": env.get("KS_BUDGET_AUTHORITY")})
        if role == "coder":
            content = candidates[min(count, len(candidates) - 1)]
        else:
            source = run.bundle.source(intent["source"]).decode()
            location = f"kernel.py:L1-L{len(source.splitlines())}"
            evidence = [{"location": location, "quote": "\n".join(source.splitlines())}]
            status = reviews[min(count, len(reviews) - 1)]
            content = {"items": [{"id": item["id"], "status": status, "reason": "synthetic independent review",
                "source_locations": [location], "evidence": evidence} for item in fidelity.obligations(run)],
                "coverage": [{"coverage_id": cid, "node": b["node"], "variant": b["variant"], "status": "pass",
                    "reason": "synthetic full coverage", "source_locations": [location], "evidence": evidence}
                    for b in run.objects["scope"]["bindings"] for cid in b["coverage_ids"]]}
            if any(item["id"] == "delivery/native_production_policy" for item in payload["materials"]):
                categories = {"framework_compliance": "framework_compliant", "compute_provenance": "self_authored_compute",
                    "dependency_inventory": "inventory_complete", "external_code_loading": "no_external_code",
                    "solution_manifest": "manifest_consistent"}
                content = {"schema_version": 2, "verdict": "allow", "summary": "explicit synthetic policy review",
                    "checks": [{"id": name, "decision": "allow", "category": category,
                        "reason": "synthetic only", "evidence": ["candidate/kernel.py:1"]} for name, category in categories.items()],
                    "dependencies": []}
        answer = {"protocol": "ks.claude.output.v1", "role": role, "assignment_sha256": payload["assignment_sha256"],
            "status": "ok", "kind": "candidate_source" if role == "coder" else "review", "content": content, "reason": "A5 synthetic only"}
        events = [{"type": "system", "subtype": "init", "tools": [], "mcp_servers": [], "skills": [], "plugins": [],
            "permissionMode": "dontAsk", "cwd": str(cwd), "session_id": session, "model": "synthetic-model"},
            {"type": "result", "subtype": "success", "is_error": False, "session_id": session, "result": json.dumps(answer),
             "usage": {"input_tokens": 5, "output_tokens": 4, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}]
        if unknown_tokens:
            events[-1].pop("usage")
        return "\n".join(json.dumps(event) for event in events), "", 0, False
    return invoke


def setup(fixed, tmp_path, *, mode="code", reviews=("pass",), candidates=(CODE,), services=None, continuing=False, checkpoint=None, prefetch=False, production=False):
    if not continuing:
        p = Publication(fixed.run.root)
        method = fidelity.freeze_method(p)
        rules = p.put_json("a5-rules.json", {"protocol": PROTOCOL, "kind": "rules", "files": [method["file"]], "tools": [method["tool"]]})
        seed = p.put_source("verify-seed", CODE.encode()) if mode == "verify_only" else None
        q = {**fixed.run.request, "mode": mode, "rules": rules, "seed": seed,
            "budget": {"parent": None, "limits": {"calls": {"coder": 4, "reviewer": 8, "planner": 0, "extract": 0},
                "evaluator_jobs": 10, "active_ms": 120000, "tokens": None}}}
        ref = p.put_json("a5-request.json", q)
        p.commit([ref])
        durable_write(fixed.run.root / "a5-request.json", jcs(q))
        fixed.a5 = prepare_run("implement", request=fixed.run.root / "a5-request.json", out=tmp_path / "a5",
                              backend_identity={"test": "A5 synthetic native services only"})
        fixed.trace = []
    run = fixed.a5
    modules = native_modules(run.objects["profile"])
    modules.agent.close()
    from dataclasses import replace
    from kernelschema.runtime import ClaudeProfile
    agent = native_agent(ClaudeProfile(**run.objects["profile"]["parameters"]["claude"]),
                         process_runner=model(run, fixed.trace, reviews=reviews, candidates=candidates))
    modules = replace(modules, agent=agent)
    service = services or FakeServices(tmp_path / "fake-jobs")
    workflow = FixedWorkflow(None, modules, context=run, supplied_services=service)
    request = native_request(run)
    if prefetch:
        request = replace(request, settings=replace(request.settings, framework_baseline_reviewers=("codex",)))
    if production:
        request = replace(request, settings=replace(request.settings, optimization_mode="production"))
    campaign = workflow.campaign(request)
    if checkpoint:
        workflow.controller.tx.checkpoint = checkpoint
    return workflow, campaign, service


def test_qualified_unpromoted_slow_C_completes_real_native_loop(fixed, tmp_path):
    workflow, campaign, service = setup(fixed, tmp_path)
    try:
        result = campaign.run()
        assert result["status"] == "completed"
        assert result["code"]["latest"] == result["code"]["qualified"]
        assert result["assessments"][-1]["performance_goal"] == "unmet"
        assert result["extensions"]["information"]["promotions"] == 0
        assert [i["role"] for i in fixed.trace] == ["coder", "reviewer"]
        assert result["usage"]["calls"]["coder"] == 1
        assert result["usage"]["evaluator_jobs"] == 2  # V0 remains charged.
        assert fixed.a5.bundle.source(result["code"]["latest"]) == CODE.encode()
    finally:
        workflow.close()
        workflow.modules.agent.close()


def test_verify_only_has_no_native_episode_or_coder(fixed, tmp_path):
    workflow, campaign, service = setup(fixed, tmp_path, mode="verify_only")
    try:
        result = campaign.run()
        assert result["status"] == "completed"
        assert result["usage"]["calls"]["coder"] == 0
        assert result["usage"]["evaluator_jobs"] == 1
        assert result["extensions"]["information"]["coder_episodes"] == 0
        assert not (campaign.workspace / ".git").exists()
    finally:
        workflow.close()
        workflow.modules.agent.close()


def test_same_R_exact_unpromoted_C_continues_across_native_episodes(fixed, tmp_path):
    first = CODE + "# first \u03b1\r\n"
    second = CODE + "# second, no final LF"
    service = FakeServices(tmp_path / "fake-jobs")
    service.comparison = FakeComparison(tmp_path / "fake-comparison")
    workflow, campaign, _ = setup(fixed, tmp_path, reviews=("fail", "pass"), candidates=(first, second), services=service)
    try:
        result = campaign.run()
        assert result["status"] == "completed"
        coders = [row for row in fixed.trace if row["role"] == "coder"]
        assert len(coders) == 2
        assert fixed.a5.bundle.source(coders[1]["source"]) == first.encode()
        assert coders[0]["input"]["assignment_sha256"] == coders[1]["input"]["assignment_sha256"]
        assert fixed.a5.bundle.source(result["code"]["latest"]) == second.encode()
        assert result["usage"]["evaluator_jobs"] == 4
        assert len(service.comparison.submissions) == 1  # Failed fidelity never reached ABBA.
    finally:
        workflow.close()
        workflow.modules.agent.close()


@pytest.mark.parametrize("boundary", ["native_invocation_retained", "candidate_sealed", "qualification_durable", "native_handoff_ready", "episode_settled"])
def test_first_result_missing_recovers_original_native_operation(fixed, tmp_path, boundary):
    fired = []
    def crash(label):
        if label == boundary and not fired:
            fired.append(label)
            raise KeyboardInterrupt("synthetic durable boundary loss")
    workflow, campaign, service = setup(fixed, tmp_path, checkpoint=crash)
    try:
        with pytest.raises(KeyboardInterrupt):
            campaign.run()
    finally:
        workflow.close()
        workflow.modules.agent.close()
    assert not (fixed.a5.root / "result.json").exists()
    restarted, campaign, _ = setup(fixed, tmp_path, services=service, continuing=True)
    try:
        result = campaign.run()
        assert result["status"] == "completed"
        assert [row["role"] for row in fixed.trace] == ["coder", "reviewer"]
        assert result["usage"]["calls"]["coder"] == 1
        assert result["usage"]["calls"]["reviewer"] == 1
        assert result["usage"]["evaluator_jobs"] == 2
        assert campaign.run() == result
    finally:
        restarted.close()
        restarted.modules.agent.close()


@pytest.mark.parametrize("after_commit", [False, True])
def test_original_ABBA_promotion_and_commit_bound_audit_recovery(fixed, tmp_path, monkeypatch, after_commit):
    service = FakeServices(tmp_path / "fake-jobs")
    service.comparison = FakeComparison(tmp_path / "fake-comparison")
    workflow, campaign, _ = setup(fixed, tmp_path, services=service)
    original = campaign.candidate_workspace.promote
    if after_commit:
        def interrupted(*args, **kwargs):
            original(*args, **kwargs)
            raise KeyboardInterrupt("synthetic loss after native promotion commit")
        monkeypatch.setattr(campaign.candidate_workspace, "promote", interrupted)
    try:
        if after_commit:
            with pytest.raises(KeyboardInterrupt):
                campaign.run()
        else:
            result = campaign.run()
    finally:
        workflow.close()
        workflow.modules.agent.close()
    if after_commit:
        workflow, campaign, _ = setup(fixed, tmp_path, services=service, continuing=True)
        try:
            result = campaign.run()
        finally:
            workflow.close()
            workflow.modules.agent.close()
    assert result["status"] == "completed"
    assert result["extensions"]["information"]["promotions"] == 1
    assert result["usage"]["evaluator_jobs"] == 3
    assert len(service.comparison.submissions) == 1
    assert len(fixed.trace) == 2


def test_V0_prefetch_group_retains_every_charge(fixed, tmp_path):
    workflow, campaign, _ = setup(fixed, tmp_path, prefetch=True)
    try:
        result = campaign.run()
        assert result["status"] == "completed"
        assert result["usage"]["calls"]["reviewer"] == 2
        assert result["usage"]["evaluator_jobs"] == 2
        ops = workflow.controller.tx.owner.operations(fixed.a5.run_id)
        members = [op for op in ops.values() if op.get("group")]
        assert len(members) == 2 and all(op["terminated"] for op in members)
    finally:
        workflow.close()
        workflow.modules.agent.close()


def test_commit_without_schema_qualification_preserves_native_history_and_stops(fixed, tmp_path, monkeypatch):
    from aka.workflow.light.long_horizon import promotion_audit
    service = FakeServices(tmp_path / "fake-jobs")
    service.comparison = FakeComparison(tmp_path / "fake-comparison")
    workflow, campaign, _ = setup(fixed, tmp_path, services=service)
    write = promotion_audit.write_promotion_audit
    def legacy(workspace, episode, evidence):
        return write(workspace, episode, {k: v for k, v in evidence.items() if k != "schema_qualification"})
    monkeypatch.setattr(promotion_audit, "write_promotion_audit", legacy)
    promote = campaign.candidate_workspace.promote
    def interrupted(*args, **kwargs):
        promote(*args, **kwargs)
        raise KeyboardInterrupt("committed native history lacks schema audit")
    monkeypatch.setattr(campaign.candidate_workspace, "promote", interrupted)
    try:
        with pytest.raises(KeyboardInterrupt):
            campaign.run()
        committed = campaign.candidate_workspace.current_revision(campaign.workspace)
    finally:
        workflow.close()
        workflow.modules.agent.close()

    workflow, campaign, _ = setup(fixed, tmp_path, services=service, continuing=True)
    try:
        result = campaign.run()
        assert result["status"] == "blocked"
        assert "lacks its original Schema qualification" in result["stop_reason"]
        assert campaign.candidate_workspace.current_revision(campaign.workspace) == committed
        assert result["extensions"]["information"]["promotions"] == 1
        assert len(fixed.trace) == 2 and len(service.comparison.submissions) == 1
    finally:
        workflow.close()
        workflow.modules.agent.close()


def test_missing_real_qualification_is_zero_fee_blocked(fixed, tmp_path):
    from aka.workflow.schema.backend import LocalServices
    workflow, campaign, _ = setup(fixed, tmp_path, services=LocalServices(tmp_path / "cpu-diagnostics"))
    try:
        result = campaign.run()
        assert result["status"] == "blocked"
        assert result["usage"]["calls"]["coder"] == result["usage"]["evaluator_jobs"] == 0
        assert not fixed.trace and not (campaign.workspace / ".git").exists()
    finally:
        workflow.close()
        workflow.modules.agent.close()


def test_owner_pause_before_V0_starts_no_paid_work(fixed, tmp_path):
    workflow, campaign, service = setup(fixed, tmp_path)
    try:
        workflow.controller.tx.owner.pause(True)
        result = campaign.run()
        assert result["status"] == "partial"
        assert result["usage"]["evaluator_jobs"] == result["usage"]["calls"]["coder"] == 0
        assert not fixed.trace and not service.evaluation.executor.submissions
    finally:
        workflow.close()
        workflow.modules.agent.close()


def test_unknown_native_return_reopens_original_locator_without_replay(fixed, tmp_path):
    workflow, campaign, service = setup(fixed, tmp_path)
    def lost(*args, **kwargs):
        raise RuntimeError("synthetic unknown original native process")
    workflow.modules.agent._process_runner = lost
    try:
        with pytest.raises(RuntimeError, match="unknown original"):
            campaign.run()
        operation = workflow.controller.tx.state["operations"]["e1-coder"]
        assert operation["locator"]["output_directory"]
    finally:
        workflow.close()
        workflow.modules.agent.close()

    workflow, campaign, _ = setup(fixed, tmp_path, services=service, continuing=True)
    try:
        result = campaign.run()
        assert result["status"] == "blocked" and result["recovery"] == "settle_only"
        assert not fixed.trace
        assert result["usage"]["calls"]["coder"] is None
        assert len(service.evaluation.executor.submissions) == 1
    finally:
        workflow.close()
        workflow.modules.agent.close()


def test_KS_external_entry_and_fresh_process_share_original_authority(fixed, tmp_path, monkeypatch):
    from kernelschema import api
    from kernelschema.budget import OwnerAccount
    from aka.workflow.schema import backend
    from aka.workflow.schema.session import native_capabilities
    from kernelschema.runtime import ClaudeProfile
    from dataclasses import replace
    import os
    import subprocess
    import sys
    workflow, campaign, service = setup(fixed, tmp_path)
    workflow.close()
    workflow.modules.agent.close()
    publication = Publication(fixed.a5.root)
    profile = {**fixed.a5.objects["profile"], "name": "aka", "capabilities": native_capabilities()}
    profile_ref = publication.put_json("external-profile.json", profile)
    request = {**fixed.a5.request, "backend": {"name": "aka", "profile": profile_ref}}
    ref = publication.put_json("external-request.json", request)
    publication.commit([ref])
    durable_write(fixed.a5.root / "external-request.json", jcs(request))
    target = tmp_path / "external-run"
    original_modules = backend.native_modules
    def selected(profile):
        modules = original_modules(profile)
        modules.agent.close()
        def process(*args, **kwargs):
            return model(open_run("implement", target), fixed.trace)(*args, **kwargs)
        agent = native_agent(ClaudeProfile(**profile["parameters"]["claude"]), process_runner=process)
        return replace(modules, agent=agent)
    monkeypatch.setattr(backend, "native_modules", selected)
    monkeypatch.setattr(backend, "services", lambda *args: service)
    result = api.implement(request=fixed.a5.root / "external-request.json", out=target)
    assert result["status"] == "completed" and result["usage"]["evaluator_jobs"] == 2
    context = open_run("implement", target)
    owner = OwnerAccount.for_context(context, create=False)
    assert all(row["authority"] == str(owner.path) for row in fixed.trace)
    script = "from kernelschema.api import open_run; from kernelschema.budget import OwnerAccount; import sys; c=open_run('implement',sys.argv[1]); o=OwnerAccount.for_context(c,create=False); print(o.path); print(len(o.operations(c.run_id)))"
    child = subprocess.run([sys.executable, "-c", script, str(target)], check=True, capture_output=True, text=True,
        env={**os.environ, "KS_BUDGET_AUTHORITY": str(owner.path)})
    assert child.stdout.splitlines() == [str(owner.path), "4"]
    replay = api.implement(continue_run=target)
    assert replay == result and len(fixed.trace) == 2


def test_slower_native_ABBA_candidate_can_still_complete(fixed, tmp_path):
    service = FakeServices(tmp_path / "fake-jobs")
    service.comparison = FakeComparison(tmp_path / "fake-comparison", candidate_latency=20)
    workflow, campaign, _ = setup(fixed, tmp_path, services=service)
    try:
        result = campaign.run()
        assert result["status"] == "completed" and result["extensions"]["information"]["promotions"] == 0
        attempt = json.loads(workflow.controller.tx.state["episodes"]["1"]["attempt"]["native_json"])
        assert attempt["verification"]["candidate_latency_us"] > attempt["verification"]["incumbent_latency_us"]
        assert result["assessments"][-1]["performance_goal"] == "unmet"
    finally:
        workflow.close()
        workflow.modules.agent.close()


def test_late_unknown_statistics_are_not_zero_or_a_second_call(fixed, tmp_path):
    workflow, campaign, _ = setup(fixed, tmp_path)
    try:
        result = campaign.run()
        tx = workflow.controller.tx
        op = tx.state["operations"]["e1-coder"]
        original = op["terminal"]["usage"]
        for sequence, tokens in ((2, None), (3, 10)):
            p = Publication(fixed.a5.root)
            ref = p.put_bytes("late-native-statistic.json", jcs({"synthetic": True, "tokens": tokens, "sequence": sequence}))
            p.commit([ref])
            tx.absorb(op, {**original, "sequence": sequence, "tokens": tokens, "receipt": ref, "receipt_id": ref["sha256"]})
            updated = campaign.run()
            assert updated["status"] == "completed"
            assert updated["usage"]["calls"]["coder"] == 1 and len(fixed.trace) == 2
            assert updated["usage"]["tokens"] == (None if tokens is None else 19)
            assert not updated["usage"]["unsettled"]
    finally:
        workflow.close()
        workflow.modules.agent.close()


def test_native_production_and_finish_review_receipts_are_paid_once(fixed, tmp_path):
    workflow, campaign, service = setup(fixed, tmp_path, production=True)
    try:
        result = campaign.run()
        operations = workflow.controller.tx.state["operations"]
        planned = [op for key, op in operations.items() if key.startswith("production-")]
        paid = [op for op in planned if op["terminal"]]
        assert paid and all(op["terminal"] or op["reservation"] is None for op in planned)
        assert result["usage"]["calls"]["reviewer"] == 1 + len(paid)
        assert workflow.controller.tx.state["finished"] is True
        assert campaign.run() == result
        assert result["usage"]["calls"]["reviewer"] == len([i for i in fixed.trace if i["role"] == "reviewer"])
    finally:
        workflow.close()
        workflow.modules.agent.close()


def test_finish_recovers_retained_native_review_before_publishing(fixed, tmp_path):
    workflow, campaign, service = setup(fixed, tmp_path, production=True)
    controller = workflow.controller
    fired = []
    def crash(boundary):
        if boundary == "native_invocation_retained" and controller.tx.state["phase"] != "running" and not fired:
            fired.append(boundary)
            raise KeyboardInterrupt("finish review returned; absorption not yet durable")
    controller.tx.checkpoint = crash
    try:
        with pytest.raises(KeyboardInterrupt):
            campaign.run()
        assert fired and not controller.tx.state["finished"]
        calls_before = len(fixed.trace)
    finally:
        workflow.close()
        workflow.modules.agent.close()
    workflow, campaign, _ = setup(fixed, tmp_path, production=True, services=service, continuing=True)
    try:
        result = campaign.run()
        assert workflow.controller.tx.state["finished"] is True
        assert not result["usage"]["unsettled"]
        assert result["status"] == "completed"
        assert len(fixed.trace) == calls_before
        assert result["usage"]["calls"]["reviewer"] == len([i for i in fixed.trace if i["role"] == "reviewer"])
        assert campaign.run() == result
    finally:
        workflow.close()
        workflow.modules.agent.close()


def test_output_limit_recovery_is_a_new_paid_call_from_exact_stable_C(fixed, tmp_path):
    workflow, campaign, _ = setup(fixed, tmp_path)
    original = workflow.modules.agent._process_runner
    count = []
    def limited(*args, **kwargs):
        stdout, stderr, code, timed_out = original(*args, **kwargs)
        if not count:
            count.append(1)
            stdout += "\n" + "x" * (workflow.controller.calls.profile.max_output_bytes + 1)
        return stdout, stderr, code, timed_out
    workflow.modules.agent._process_runner = limited
    try:
        result = campaign.run()
        assert result["status"] == "completed"
        assert result["usage"]["calls"]["coder"] == 2
        assert result["usage"]["calls"]["reviewer"] == 1
        coders = [row for row in fixed.trace if row["role"] == "coder"]
        assert coders[0]["source"]["sha256"] == coders[1]["source"]["sha256"]
        assert fixed.a5.bundle.source(coders[0]["source"]) == fixed.a5.bundle.source(coders[1]["source"])
        assert coders[0]["session"] != coders[1]["session"]
    finally:
        workflow.close()
        workflow.modules.agent.close()


def test_fresh_child_consumes_original_reservation_without_copying_account(fixed, tmp_path):
    import os
    import subprocess
    import sys
    workflow, campaign, _ = setup(fixed, tmp_path)
    tx = workflow.controller.tx
    op = tx.plan("external-synthetic-review", role="reviewer", source=tx.state["latest"])
    tx.reserve(op)
    tx.intent(op, {"synthetic_child": "same original account"})
    script = """
import json, sys
from kernelschema.api import open_run
from kernelschema.budget import OwnerAccount
c = open_run('implement', sys.argv[1])
o = OwnerAccount.for_context(c, create=False)
i = next(i for i in c.intents() if i['operation_id'] == sys.argv[2])
assert o.authorize(i['reservation'])
row = dict(producer='A5-synthetic-external-child', operation_id=i['operation_id'], role=i['role'],
           purpose=i['purpose'], receipt_id='synthetic-child-receipt', sequence=1, receipt=None,
           state='settled', calls=1, evaluator_jobs=0, tokens=None, gpu_ms=0)
assert o.absorb(c.run_id, row, terminated=True)
assert not o.absorb(c.run_id, row, terminated=True)
print(o.path)
"""
    try:
        child = subprocess.run([sys.executable, "-I", "-c", script, str(fixed.a5.root), op["id"]],
            env={**os.environ, "KS_BUDGET_AUTHORITY": str(tx.owner.path)}, check=True, capture_output=True, text=True)
        assert child.stdout.strip() == str(tx.owner.path)
        owned = tx.owner.operations(fixed.a5.run_id)[op["id"]]
        assert owned["terminated"] and owned["latest"]["calls"] == 1
        assert owned["latest"]["tokens"] is None and len(owned["history"]) == 1
        assert not tx.owner.fences()
    finally:
        workflow.close()
        workflow.modules.agent.close()


@pytest.mark.parametrize("command", ["workflow", "optimize"])
def test_actual_CLI_schema_default_factory_returns_zero_fee_blocked_result(fixed, tmp_path, command):
    import subprocess
    import sys
    from aka.workflow.schema import backend
    workflow, campaign, _ = setup(fixed, tmp_path)
    workflow.close()
    workflow.modules.agent.close()
    config = tmp_path / "selection.json"
    root = tmp_path / "cli-schema-run"
    config.write_bytes(jcs({"format": "aka.schema.selection.v1", "request": str(fixed.a5.root / "request.json"),
        "run_root": str(root), "backend": {"factory": "aka.workflow.schema.backend:create_workflow",
            "implementation_sha256": sha256(Path(backend.__file__).read_bytes()), "parameters": {}}}))
    repo = Path(__file__).resolve().parents[1]
    child = subprocess.run([sys.executable, "-I", "-m", "aka", command, "--repo-root", str(repo),
        "--problem", str(fixed.a5.root), "--workspace-base", str(tmp_path / "native-cli"),
        "--platform", "H20", "--framework", "triton", "--framework-baseline", "never", "--convert-after", "0",
        "--schema", "--schema-config", str(config)], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    result = loads((root / "result.json").read_bytes())
    assert result["status"] == "blocked"
    assert result["usage"]["evaluator_jobs"] == result["usage"]["calls"]["coder"] == 0


def test_settled_V0_and_prefetch_survive_lost_native_return(fixed, tmp_path):
    workflow, campaign, service = setup(fixed, tmp_path, prefetch=True)
    original = campaign.measure_for_acceptance
    def lost(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt("V0 Task is settled but native return was lost")
    campaign.measure_for_acceptance = lost
    try:
        with pytest.raises(KeyboardInterrupt):
            campaign.run()
        assert len(service.evaluation.executor.submissions) == 1
    finally:
        workflow.close()
        workflow.modules.agent.close()

    workflow, campaign, _ = setup(fixed, tmp_path, services=service, continuing=True, prefetch=True)
    try:
        result = campaign.run()
        assert result["status"] == "completed"
        assert result["usage"]["evaluator_jobs"] == 2
        assert result["usage"]["calls"]["coder"] == 1
        assert result["usage"]["calls"]["reviewer"] == 2
        assert len(service.evaluation.executor.submissions) == 2
    finally:
        workflow.close()
        workflow.modules.agent.close()


def test_AKA_selection_bootstrap_gap_does_not_create_a_replacement_owner(fixed, tmp_path):
    from aka.workflow.schema import backend
    from aka.workflow.schema.provider import SchemaSelection
    from kernelschema.contracts import ProtocolError
    workflow, _, _ = setup(fixed, tmp_path)
    modules = workflow.modules
    workflow.close()
    config = tmp_path / "bootstrap-selection.json"
    root = tmp_path / "bootstrap-run"
    config.write_bytes(jcs({"format": "aka.schema.selection.v1", "request": str(fixed.a5.root / "request.json"),
        "run_root": str(root), "backend": {"factory": "aka.workflow.schema.backend:create_workflow",
            "implementation_sha256": sha256(Path(backend.__file__).read_bytes()), "parameters": {}}}))
    first = backend.create_workflow(SchemaSelection.load(config), modules)
    first.close()  # Original RunContext exists, but no account binding was made.
    reopened = backend.create_workflow(SchemaSelection.load(config), modules)
    try:
        with pytest.raises(ProtocolError, match="owner binding missing"):
            reopened.campaign(native_request(reopened.context))
        assert not (root / ".runtime/state/budget.json").exists()
        assert not (root / ".runtime/owner-account.json").exists()
    finally:
        reopened.close()
        modules.agent.close()
