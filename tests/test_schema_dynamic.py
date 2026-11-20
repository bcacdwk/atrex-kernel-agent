"""A6 CPU: real public KS API/native Campaign, explicitly synthetic transports."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from kernelschema import api, fidelity, template_api, writing
from kernelschema.artifacts import Bundle, Publication, durable_write
from kernelschema.budget import OwnerAccount, cost
from kernelschema.contracts import PROTOCOL, ProtocolError, digest, jcs, loads, sha256
from kernelschema.resources import checkout_root
from kernelschema.runtime import ClaudeProfile
from aka.contracts.workflow import ProblemSource
from aka.workflow.legacy.public_state import campaign_lease
from aka.workflow.schema import backend
from aka.workflow.schema.dynamic import DynamicWorkflow
from aka.workflow.schema.provider import SchemaSelection
from aka.workflow.schema.session import native_agent, native_capabilities
from test_schema_optional_module import fixed
from test_schema_backend import CODE, FakeServices, model


@pytest.fixture
def dynamic(fixed, tmp_path, monkeypatch):
    def make(*, start="schema", action="none", decision="keep", optimize=True,
             reviews=("pass",), unknown_tokens=False, mode="code", lose_native=False, planner_limit=2, production=False,
             fixed_limits=None, fixed_native_limits=None):
        root = fixed.run.root
        p = Publication(root)
        owner = OwnerAccount.create(tmp_path / "owner.json", owner="synthetic-a6-parent",
            limits={"calls": {"planner": planner_limit, "reviewer": 16, "coder": 8, "extract": 0},
                    "evaluator_jobs": 16, "active_ms": 600000, "tokens": None})
        parent = p.put_json("a6-parent.json", owner.descriptor)
        rules = writing.freeze_rules(p)
        rules_doc = loads((p.stage / rules["path"]).read_bytes())
        method = fidelity.freeze_method(p)
        rules = p.put_json("a6-rules.json", {**rules_doc, "files": sorted([*rules_doc["files"], method["file"]], key=lambda r: r["path"]),
                                            "tools": [*rules_doc["tools"], method["tool"]]})
        old = fixed.run.objects["profile"]
        if fixed_native_limits is not None:
            old = {**old, "native_limits": {**old["native_limits"], "workflow_limits": fixed_native_limits}}
        profile = p.put_json("a6-native-profile.json", {**old, "name": "aka", "capabilities": native_capabilities()})
        limits = {"calls": {"planner": 0, "reviewer": 8, "coder": 4, "extract": 0},
                  "evaluator_jobs": 8, "active_ms": 120000, "tokens": None}
        implementation_limits = fixed_limits if fixed_limits is not None else limits
        seed = p.put_source("a6-explicit-seed", CODE.encode()) if mode == "verify_only" else None
        q = {**fixed.run.request, "request_id": "a6-template", "rules": rules, "mode": mode, "seed": seed,
             "budget": {"parent": parent, "limits": implementation_limits}, "backend": {"name": "aka", "profile": profile}}
        qref = p.put_json("a6-template.json", q)
        p.commit([qref])
        durable_write(root / "a6-template.json", jcs(q))
        fake_dirs = []
        def writing_request(operation):
            fake = tmp_path / ("fake-" + operation)
            fake.mkdir()
            script = checkout_root() / "tests/workflows/fixtures/writing/fake_claude.py"
            executable = fake / "claude"
            executable.write_text("#!" + sys.executable + "\n" + "\n".join(script.read_text().splitlines()[1:]) + "\n")
            executable.chmod(0o755)
            runtime = ClaudeProfile.inspect(requested_model="synthetic-model", executable=str(executable), timeout_ms=5000)
            doc = template_api.effective_document(Bundle(root), Bundle(root).document(q["schema"]), "root")
            chosen = "revise" if operation == "design" else decision
            if operation == "design":
                doc["CONTRACT"] = writing.template_contract(Bundle(root), q["task"])
            elif chosen == "revise":
                doc["DESIGN"]["Work_schedule"]["Step"]["MAIN"]["Schedule"]["Worker_assignment"]["Preassigned"]["count"] = 2
            scope_doc = Bundle(root).document(q["scope"])
            draft = {"format": "ks.writing.draft.v1", "decision": chosen, "reason": "SYNTHETIC A6 only",
                     "selection": [{"id": "workflow_objective", "reason": "authorized synthetic goal"}],
                     "raw": [] if chosen == "keep" else [{"node": "root", "text": yaml.safe_dump(doc, sort_keys=False, width=1000)}],
                     "dispatch": None, "bindings": None if chosen == "keep" else scope_doc["bindings"],
                     "next_action": "none" if operation == "design" else action,
                     "action_reason": "synthetic authorized followup" if operation == "optimize" and action != "none" else None,
                     "action_inputs": ["evidence:0" if action == "verify" else "task/default_start"] if operation == "optimize" and action != "none" else [],
                     "action_evidence": ["material:workflow_objective"] if operation == "optimize" and action != "none" else []}
            (fake / "scenario.json").write_text(json.dumps({"draft": draft, "review": "pass"}))
            fake_dirs.append(fake)
            pub = Publication(root)
            business = writing.backend_profile(runtime, fixed.task["environment"] if "environment" in fixed.task else fixed.run.objects["task"]["environment"])
            review_count = 4 if production else 1
            a = cost(role="coder", calls=1, evaluator_jobs=2, active_ms=3000 + review_count * 1000)
            b = cost(role="reviewer", calls=review_count)
            business["parameters"]["writing_finish_floor"] = {key: a[key] + b[key] for key in a}
            pref = pub.put_json(operation + "-profile.json", business)
            scope = pub.put_json("a6-design-scope.json", {**scope_doc, "bindings": []}) if operation == "design" else q["scope"]
            template_operation = "design" if start == "design" else operation
            w = writing.request(operation=template_operation, request_id="synthetic-a6-" + operation,
                task=q["task"], rules=rules, materials=q["materials"], scope=scope,
                budget={"parent": parent, "limits": {**limits, "calls": {**limits["calls"], "planner": 1}}},
                profile=pref, **({"schema": q["schema"]} if template_operation == "optimize" else {}))
            ref = pub.put_json(operation + "-request.json", w)
            pub.commit([ref])
            path = root / (operation + "-request.json")
            durable_write(path, jcs(w))
            return {"path": str(path), "sha256": sha256(path.read_bytes())}
        initial = writing_request("design")["path"] if start == "design" else str(root / "a6-template.json")
        config = {"start": start, "authority": str(owner.path),
                  "implementation": {"profile": profile, **{k: q[k] for k in ("rules", "materials", "budget", "mode", "seed")}} if start == "design" else None,
                  "optimize_request": writing_request("optimize") if optimize else None}
        selection_path = tmp_path / "selection.json"
        selection_path.write_bytes(jcs({"format": "aka.schema.selection.v1", "request": initial,
            "run_root": str(tmp_path / "dynamic"), "backend": {"factory": "aka.workflow.schema.backend:create_workflow",
            "implementation_sha256": sha256(Path(backend.__file__).read_bytes()), "parameters": {"dynamic": config}}}))
        selection = SchemaSelection.load(selection_path)
        modules = backend.native_modules(Bundle(root).document(profile))
        modules.agent.close()
        trace = []
        def runner(command, **kwargs):
            if lose_native:
                raise RuntimeError("synthetic unknown original native process")
            for out in (tmp_path / "dynamic/runs").iterdir():
                if loads((out / "request.json").read_bytes())["operation"] != "implement":
                    continue
                run = api.open_run("implement", out)
                if str(out) in command[3]:
                    return model(run, trace, reviews=reviews, unknown_tokens=unknown_tokens)(command, **kwargs)
            raise AssertionError("no original native run for synthetic invocation")
        agent = native_agent(ClaudeProfile(**old["parameters"]["claude"]), process_runner=runner)
        modules = replace(modules, agent=agent)
        service = FakeServices(tmp_path / "synthetic-jobs")
        monkeypatch.setattr(backend, "services", lambda *_: service)
        workflow = backend.create_workflow(selection, modules)
        request = replace(backend.native_request(fixed.run), problem=ProblemSource(root, tmp_path / "outer-problem"),
                          workspace=tmp_path / "native", name="a6_synthetic")
        if production:
            request = replace(request, settings=replace(request.settings, optimization_mode="production"))
        return SimpleNamespace(workflow=workflow, request=request, trace=trace, owner=owner,
                               modules=modules, selection=selection, fake_dirs=fake_dirs, services=service)
    yield make


def run(d):
    with campaign_lease(d.request.workspace, writer=True):
        return d.workflow.campaign(d.request).run()


def close(d):
    d.workflow.close()
    d.modules.agent.close()
    d.modules.experiment_factory.close()


def test_existing_schema_completed_keep_stops(dynamic):
    d = dynamic()
    try:
        result = run(d)
        assert result["phase"] == "done" and result["status"] == "completed", result
        assert len(result["history"]) == 2 and result["optimize_count"] == 1
        assert len(d.trace) == 2
        assert result["current_qualified"] is not None
        before = d.owner.operations()
        assert run(d) == result
        assert d.owner.operations() == before
    finally:
        close(d)


@pytest.mark.parametrize("decision,action", [("keep", "implement"), ("keep", "verify"), ("revise", "implement")])
def test_action_really_executes_fixed_same_native_history(dynamic, decision, action):
    d = dynamic(decision=decision, action=action)
    try:
        result = run(d)
        assert result["phase"] == "done" and result["status"] == "completed", result
        assert len(result["history"]) == 3
        first, _, followup = result["history"]
        q = Bundle(d.workflow.root).document(followup["request"])
        a = Bundle(d.workflow.root).document(first["result"])
        b = Bundle(d.workflow.root).document(followup["result"])
        assert q["budget"]["parent"] == Bundle(d.workflow.root).document(first["request"])["budget"]["parent"]
        assert b["usage"]["evaluator_jobs"] == 1  # Existing V0 was not replayed.
        review = Bundle(d.workflow.root).document(b["assessments"][-1]["fidelity"]["receipt"])
        assert review["subject"]["schema_sha256"] == q["schema"]["sha256"]
        if action == "verify":
            assert q["mode"] == "verify_only" and q["seed"] == a["code"]["latest"]
            assert b["usage"]["calls"]["coder"] == 0
        else:
            assert q["seed"] is None  # Explicit Task default_start does not pick old C.
            assert b["usage"]["calls"]["coder"] == 1
            assert followup["native"]["episodes"] == 2
            task = Bundle(d.workflow.root).document(q["task"])
            coders = [row for row in d.trace if row["role"] == "coder"]
            assert coders[-1]["source"] == task["starter_source"]
        assert len({op["reservation"]["account_id"] for op in d.owner.operations().values()}) == 1
    finally:
        close(d)


def test_original_startup_prepares_and_runs_dynamic_native_stages(dynamic):
    from aka.workflow.legacy.startup import WorkflowStartup
    d = dynamic(action="implement", decision="revise")
    try:
        startup = WorkflowStartup(d.workflow)
        code = startup._run_campaign(d.request, d.workflow._load_problem(d.request))
        assert code == 0
        result = loads((d.workflow.root / "dynamic-result.json").read_bytes())
        assert result["status"] == "completed" and len(result["history"]) == 3
        assert len([row for row in d.trace if row["role"] == "coder"]) == 2
    finally:
        close(d)


def test_action_extra_inputs_are_explicitly_unsupported(dynamic):
    d = dynamic(action="implement", decision="revise")
    scenario = d.fake_dirs[0] / "scenario.json"
    value = json.loads(scenario.read_text())
    value["draft"]["action_inputs"].append("material:workflow_objective")
    scenario.write_text(json.dumps(value))
    try:
        result = run(d)
        assert result["status"] == "blocked" and result["recovery"] == "new_request"
        assert len(result["history"]) == 2 and result["optimize_count"] == 1
        assert result["current_qualified"] is None
        assert len([row for row in d.trace if row["role"] == "coder"]) == 1
        assert "supports only exact" in result["reason"]
    finally:
        close(d)


def test_real_ks_reservations_keep_followup_floor_and_single_clock(dynamic):
    d = dynamic()
    try:
        result = run(d)
        assert result["status"] == "completed"
        child = api.open_run("optimize", d.workflow.root / result["history"][-1]["out"])
        operations = d.owner.operations(child.run_id)
        assert len(operations) == 2
        for op in operations.values():
            assert op["floor"]["coder"] == 1
            assert op["floor"]["evaluator_jobs"] == 2
            assert op["floor"]["reviewer"] == (2 if op["role"] == "planner" else 1)
            assert op["floor"]["active_ms"] == (5000 if op["role"] == "planner" else 4000)
            assert len(op["history"]) == 1
        ledger = d.owner._read()  # Inspect the original ledger; no second clock.
        assert ledger["active_run"] is None
        assert int(ledger["active_ns"]) == sum(int(q["active_ns"]) for q in ledger["requests"].values())
        before = d.owner.operations()
        assert run(d) == result
        assert d.owner.operations() == before
    finally:
        close(d)


def test_fixed_role_reader_resolves_new_ks_text_alias_without_widening_grants(fixed, tmp_path):
    from threading import Event
    from kernelschema.runtime import RoleDelivery, material_text
    from aka.workflow.schema.context import RoleContext, RoleReader
    objective = fixed.run.objects["materials"]["entries"][0]["file"]
    role = RoleContext.prepare(fixed.run, "reviewer", source=fixed.task["starter"],
                              deliveries=(RoleDelivery("duplicate_objective", objective, ("reviewer",)),))
    alias = next(item for item in role.document["materials"] if item["id"] == "delivery/duplicate_objective")
    assert "text_alias_of" in alias and "text" not in alias
    reader = RoleReader(role, tmp_path)
    context = SimpleNamespace(workspace=tmp_path, cancel_event=Event())
    reply = loads(reader.handle(jcs({"op": "read", "id": alias["id"]}), context))
    assert reply["text"] == material_text(role.document, alias["id"])
    assert reply["text"].encode() == fixed.run.bundle.read(objective)
    with pytest.raises(PermissionError):
        reader.handle(jcs({"op": "read", "id": "evaluator/inputs.json"}), context)


def test_initial_design_then_native_fixed(dynamic):
    d = dynamic(start="design", optimize=False)
    try:
        result = run(d)
        assert d.selection.request_document()["operation"] == "design"
        assert "schema" not in d.selection.request_document()
        assert result["status"] == "completed", result
        assert len(result["history"]) == 2 and result["optimize_count"] == 0
        first = Bundle(d.workflow.root).document(result["history"][0]["result"])
        assert first["operation"] == "design" and first["schema"] == result["schema"]
        planner = json.loads((d.fake_dirs[0] / "invocations.jsonl").read_text().splitlines()[0])
        assert not any(m["id"].startswith("schema/") for m in planner["materials"])
    finally:
        close(d)
