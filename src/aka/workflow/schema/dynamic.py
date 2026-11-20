"""Finite AKA owner: optional design, fixed native work, one optimize/action.

Writing stays in KS's public API; fixed work stays in FixedController. This
module persists stage identities and handoffs, never a budget or role runtime.
"""
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import os
from pathlib import Path
from uuid import uuid4

from kernelschema import api, writing
from kernelschema.artifacts import Bundle, Publication, dependencies, durable_write, import_bundle, read_regular
from kernelschema.budget import BudgetDenied, OwnerAccount, cost, use_authority
from kernelschema.contracts import ProtocolError, UnsupportedError, digest, jcs, loads, sha256, validate
from kernelschema.materials import validate_request

from .backend import FixedWorkflow, implementation
from .handoff import close_stage
from .state import put_once

VERSION = "aka.dynamic.v1"


def configuration(selection):
    selected = selection.document["backend"]["parameters"].get("dynamic")
    fields = {"start", "authority", "implementation", "optimize_request"}
    if not isinstance(selected, dict) or set(selected) != fields or selected["start"] not in {"schema", "design"}:
        raise UnsupportedError("dynamic requires start, authority, implementation and optimize_request; at most one optimize")
    value = dict(selected)
    if not isinstance(value["authority"], str) or not value["authority"]:
        raise ProtocolError("dynamic requires the explicit existing parent authority")
    value["authority"] = str((selection.origin.parent / value["authority"]).resolve())
    initial = selection.request_document()
    fixed = fixed_template(selection)
    if fixed["budget"]["parent"] is None:
        raise ProtocolError("dynamic requests require the same explicit existing parent account")
    if fixed["backend"]["name"] != "aka":
        raise ProtocolError("dynamic fixed template must select backend=aka")
    parent = Bundle(Path(selection.document["request"]).parent).document(fixed["budget"]["parent"])
    if OwnerAccount(value["authority"]).descriptor != parent:
        raise ProtocolError("dynamic parent descriptor differs from original authority")
    for operation in ("optimize",):
        key = "optimize_request"
        pinned = value[key]
        if pinned is None:
            continue
        if not isinstance(pinned, dict) or set(pinned) != {"path", "sha256"}:
            raise ProtocolError("writing template requires an explicit path and sha256")
        path = (selection.origin.parent / pinned["path"]).resolve(strict=True)
        raw = path.read_bytes()
        if sha256(raw) != pinned["sha256"]:
            raise ProtocolError("original writing template bytes changed")
        q = validate(loads(raw))
        objects = validate_request(Bundle(path.parent), q)
        if q["operation"] not in {"design", "optimize"} or q["task"] != fixed["task"] or q["budget"]["parent"] != fixed["budget"]["parent"]:
            raise ProtocolError("writing template must share operation, exact Task and parent")
        scope = Bundle(Path(selection.document["request"]).parent).document(fixed["scope"])
        if objects["scope"]["coverage_ids"] != scope["coverage_ids"]:
            raise ProtocolError("writing cannot change full Task coverage")
        objectives = [e for e in objects["materials"]["entries"] if e["id"] == "workflow_objective" and
                      {"planner", "reviewer"} <= set(e["roles"])]
        if len(objectives) != 1 or not Bundle(path.parent).read(objectives[0]["file"]).strip():
            raise ProtocolError("writing requires one explicitly authorized nonempty objective")
        value[key] = {"path": str(path), "sha256": pinned["sha256"]}
    if initial["budget"]["parent"] != fixed["budget"]["parent"]:
        raise ProtocolError("initial writing and implementation must share the same parent")
    return value


def fixed_template(selection):
    """A design start carries implementation inputs, with no invented Schema."""
    initial = selection.request_document()
    selected = selection.document["backend"]["parameters"]["dynamic"]
    if selected.get("start") == "schema":
        if selected.get("implementation") is not None:
            raise ProtocolError("schema start uses its original implement request; implementation must be null")
        return initial
    settings = selected.get("implementation")
    if not isinstance(settings, dict) or set(settings) != {"profile", "rules", "materials", "budget", "mode", "seed"}:
        raise ProtocolError("design start needs explicit implementation profile/rules/materials/budget/mode/seed")
    bundle = Bundle(Path(selection.document["request"]).parent)
    for key, kind in (("profile", "backend_profile"), ("rules", "rules"), ("materials", "materials")):
        bundle.document(settings[key], kind)
    if settings["mode"] not in {"code", "verify_only"}:
        raise ProtocolError("unsupported implementation mode")
    if settings["seed"] is not None:
        bundle.source(settings["seed"])
    if settings["mode"] == "verify_only" and (settings["seed"] is None or not bundle.source(settings["seed"])):
        raise ProtocolError("verify_only requires an explicit nonempty Source")
    from kernelschema.contracts import limits
    if not isinstance(settings["budget"], dict) or set(settings["budget"]) != {"parent", "limits"}:
        raise ProtocolError("implementation needs an explicit immutable budget")
    limits(settings["budget"]["limits"])
    return {"protocol": initial["protocol"], "operation": "implement", "request_id": "aka-dynamic-fixed",
            "task": initial["task"], "schema": None, "scope": initial["scope"],
            **{k: settings[k] for k in ("rules", "materials", "budget", "mode", "seed")},
            "backend": {"name": "aka", "profile": settings["profile"]}}


class DynamicWorkflow:
    """Original WorkflowStartup holds its one writer lease for the whole chain."""
    def __init__(self, selection, modules, *, checkpoint=None):
        self.selection, self.modules = selection, modules
        self.config = configuration(selection)
        self.root = Path(selection.document["run_root"])
        self.owner = OwnerAccount(self.config["authority"])
        self.checkpoint = checkpoint or (lambda label, workflow: None)
        self.fixed = fixed_template(selection)
        self.bundle = Bundle(Path(selection.document["request"]).parent)
        self.children = []
        self.controller = None
        self.binding = {"version": VERSION, "selection": selection.identity, "configuration": self.config,
                        "initial": digest(selection.request_document()),
                        "fixed": digest(self.fixed), "implementation": implementation(self.bundle.document(self.fixed["backend"]["profile"])),
                        "owner": self.owner.descriptor}
        anchor = self.root / ".runtime/aka-dynamic-anchor.json"
        if anchor.exists():
            if loads(anchor.read_bytes()) != self.binding:
                raise ProtocolError("original dynamic implementation/selection/owner changed")
            self._reload()
        else:
            if self.root.exists():
                raise ProtocolError("existing dynamic root lacks its original binding")
            put_once(anchor, self.binding)
            import_bundle(self.bundle, self.root, [*dependencies(self.fixed), *dependencies(selection.request_document())])
            self.state = {"binding": digest(self.binding), "phase": "design" if self.config["start"] == "design" else "fixed",
                          "schema": None if self.config["start"] == "design" else self.fixed["schema"],
                          "scope": self.fixed["scope"], "current": None, "history": [], "optimize_count": 0,
                          "handoff": None, "status": "prepared", "recovery": "continue", "reason": "finite dynamic workflow prepared",
                          "action": {"mode": self.fixed["mode"], "seed": self.fixed["seed"]}, "native_request": None}
            self._save("created")
        from aka.workflow.light.provider import WorkflowOptions
        from .backend import LocalServices
        from .evaluation import SchemaExecution, SchemaEvaluatorFactory
        execution = SchemaExecution(self.root / ".runtime/launch-evaluations",
                                    LocalServices(self.root / ".runtime/launch-services").evaluation)
        def unowned(_):
            raise ProtocolError("dynamic launch has no fixed operation; admit a stage first")
        self.native = modules.light(agent=modules.agent, problem=selection.problem_provider(),
            executor=execution, evaluator_factory=SchemaEvaluatorFactory(execution=execution, operation=unowned),
            options=WorkflowOptions(file_system="single_file", output_limit_recovery_retries=0))

    def _reload(self):
        saved = loads(read_regular(self.root, ".runtime/aka-dynamic-state.json")[0])
        if set(saved) != {"state", "sha256"} or digest(saved["state"]) != saved["sha256"]:
            raise ProtocolError("original dynamic state is corrupt; cannot replay stages")
        self.state = saved["state"]
        if self.state["binding"] != digest(self.binding):
            raise ProtocolError("original dynamic state binding is missing or corrupt")
        if (self.state["phase"] not in {"design", "fixed", "optimize", "followup", "done"}
                or type(self.state["optimize_count"]) is not int or not 0 <= self.state["optimize_count"] <= 1
                or len(self.state["history"]) > 4):
            raise ProtocolError("dynamic state exceeds the finite workflow")

    def _save(self, label):
        durable_write(self.root / ".runtime/aka-dynamic-state.json",
                      jcs({"state": self.state, "sha256": digest(self.state)}), replace=True)
        self.checkpoint(label, self)

    def _load_problem(self, request):
        return self.native._load_problem(request)

    def _campaign(self, request, problem):
        campaign = self.native._campaign(request, problem)
        campaign.run = lambda: self.execute(request)
        campaign.schema_workflow = self
        return campaign

    def campaign(self, request):
        return self._campaign(request, self._load_problem(request))

    @contextmanager
    def _locked(self):
        fd = os.open(self.root / ".runtime/aka-dynamic.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            self._reload()
            yield
        finally:
            os.close(fd)

    def _request(self):
        phase = self.state["phase"]
        if phase in {"design", "optimize"}:
            path, q = self._writing_template(phase)
            import_bundle(Bundle(path.parent), self.root, dependencies(q))
            args = {k: q[k] for k in ("task", "rules", "materials", "budget")}
            args.update(operation=phase, request_id="aka-dynamic-" + phase, profile=q["backend"]["profile"],
                        scope=q["scope"] if phase == "design" else self.state["scope"])
            if phase == "optimize":
                previous = Bundle(self.root).document(self.state["history"][-1]["result"])
                evidence = list(q.get("evidence", []))
                for ref in [previous["code"]["latest"], *[a[k]["receipt"] for a in previous["assessments"]
                            for k in ("build", "correctness", "performance", "coverage", "fidelity")]]:
                    if ref and ref not in evidence:
                        evidence.append(ref)
                args.update(schema=self.state["schema"], feedback=[*q.get("feedback", []), *previous["feedback"]], evidence=evidence)
            return writing.request(**args)
        return validate({**self.fixed, "request_id": "aka-dynamic-" + phase,
                         "schema": self.state["schema"], "scope": self.state["scope"], **self.state["action"]})

    def _writing_template(self, phase):
        path = Path(self.selection.document["request"] if phase == "design" else self.config["optimize_request"]["path"])
        raw = read_regular(path.parent, path.name)[0]
        q = loads(raw)
        expected = self.binding["initial"] if phase == "design" else self.config["optimize_request"]["sha256"]
        if (digest(q) if phase == "design" else sha256(raw)) != expected:
            raise ProtocolError("original writing input changed before stage admission")
        return path, q

    def _writing_floors(self, request):
        """Require KS's actual reservations to retain the next native finish.

        This is a policy minimum, not a balance check or a new reservation.
        The sole OwnerAccount enforces it on planner and reviewer launch.
        """
        profile = self.bundle.document(self.fixed["backend"]["profile"])
        minimum = profile["parameters"].get("minimum_active_ms", {"coder": 1000, "reviewer": 1000, "evaluator": 1000})
        if set(minimum) != {"coder", "reviewer", "evaluator"} or any(type(v) is not int or v <= 0 for v in minimum.values()):
            raise ProtocolError("positive native role minimum_active_ms required")
        for phase in (["design"] if self.config["start"] == "design" else []) + (["optimize"] if self.config["optimize_request"] else []):
            path, q = self._writing_template(phase)
            business = Bundle(path.parent).document(q["backend"]["profile"])
            code = phase == "optimize" or self.fixed["mode"] == "code"
            reviews = (4 if request.settings.optimization_mode == "production" else 1)
            reviews += len(request.settings.framework_baseline_reviewers) if code else 0
            jobs = 2 if code else 1  # Worst allowed action includes a fresh V0.
            a = cost(role="coder", calls=int(code), evaluator_jobs=jobs,
                     active_ms=minimum["coder"] * int(code) + minimum["evaluator"] * jobs + minimum["reviewer"] * reviews)
            b = cost(role="reviewer", calls=reviews)
            required = {key: a[key] + b[key] for key in a}
            floor = business["parameters"].get("writing_finish_floor")
            if (not isinstance(floor, dict) or set(floor) != set(required)
                    or any(type(floor[k]) is not int or floor[k] < required[k] for k in required)):
                raise UnsupportedError("dynamic writing profile must freeze writing_finish_floor covering the next native stage: " + str(required))
            # KS applies the writing floor to the writing request and shared
            # parent at each actual reservation. The next fixed request has its
            # own immutable request/native ceilings; check those static policy
            # ceilings before spending on a design it could never implement.
            # This neither computes remaining credit nor reserves a fake call.
            for name, limits in (("request", self.fixed["budget"]["limits"]),
                                 ("native", profile["native_limits"].get("workflow_limits"))):
                if limits is None:
                    continue
                from kernelschema.contracts import limits as validate_limits
                validate_limits(limits)
                for key, amount in required.items():
                    maximum = limits["calls"].get(key) if key in {"planner", "coder", "reviewer", "extract"} else limits.get(key)
                    if maximum is not None and maximum < amount:
                        raise UnsupportedError("next fixed " + name + " cannot cover minimum finish " + key)
                if limits.get("tokens") is not None:
                    raise UnsupportedError("next fixed " + name + " cannot enforce a finite native token budget")

    def _ready(self):
        if self.owner.fences():
            raise BudgetDenied("original owner operation unresolved; settle original stage")
        q = self._request()
        validate_request(Bundle(self.root), q)
        stage = "stage-" + str(len(self.state["history"]))
        current = {"operation": q["operation"], "phase": self.state["phase"], "out": "runs/" + stage,
                   "request_path": stage + ".request.json", "publication": uuid4().hex}
        # Persist publication identity before any run/request publication.
        self.state["current"] = {**current, "document": q, "request": None}
        self._save("new_request_intent")

    def _publish_request(self):
        current = self.state["current"]
        if current["request"] is None:
            p = Publication(self.root, current["publication"])
            current["request"] = p.put_json("request.json", current["document"])
            p.commit([current["request"]])
            put_once(self.root / current["request_path"], current["document"])
            self._save("new_request_ready")

    def _fixed(self, request):
        current = self.state["current"]
        q, out = current["document"], self.root / current["out"]
        profile = Bundle(self.root).document(q["backend"]["profile"])
        identity = implementation(profile)
        continuing = out.exists()
        context = api.open_run("implement", out, backend_identity=identity) if continuing else api.prepare_run(
            "implement", request=self.root / current["request_path"], out=out, backend_identity=identity)
        # An existing run without budget binding is ambiguous and never creates
        # an account. Only a genuinely new prepare attaches the selected parent.
        OwnerAccount.for_context(context, create=not continuing)
        self.checkpoint("new_request_bound", self)
        from aka.contracts.workflow import ProblemSource
        native_request = replace(request, problem=ProblemSource(context.root, context.root / ".runtime/aka-problem"))
        child = FixedWorkflow(None, self.modules, context=context, continuing=continuing,
                              handoff=self.state["handoff"] if q["mode"] == "code" else None)
        self.children.append(child)
        snapshot = native_request.problem.snapshot_directory
        if self.state["handoff"] and not snapshot.exists():
            # Light's ordinary reopen forbids inventing a missing Task snapshot.
            # This explicit new-Q boundary captures a *new* scope snapshot only
            # before its first intent. It never repairs a lost active snapshot.
            if context.intents() or (context.root / ".runtime/state/aka-source-lifecycle.json").exists():
                raise ProtocolError("active stage lost its original Task snapshot")
            problem = child.native.problem.load(context.root,
                optimization_mode=request.settings.optimization_mode, snapshot_directory=snapshot,
                expose_evaluator_shapes=request.settings.expose_evaluator_shapes)
            campaign = child._campaign(native_request, problem)
            from .handoff import validate_handoff
            validate_handoff(self.state["handoff"], context, campaign)
            self.checkpoint("new_scope_snapshot_ready", self)
        else:
            campaign = child.campaign(native_request)
        self.controller = child.controller
        self.checkpoint("new_e_unadmitted", self)
        campaign.prepare()
        result = campaign.run()
        return context, result, campaign

    def _writing(self):
        current = self.state["current"]
        out = self.root / current["out"]
        operation = getattr(api, current["operation"])
        result = operation(continue_run=out) if out.exists() else operation(
            request=self.root / current["request_path"], out=out)
        return api.open_run(current["operation"], out), result, None

    def _action(self, result):
        if not result["action_evidence"] or result["action_inputs"][:2] != [result["schema"], result["scope"]]:
            raise ProtocolError("writing action lacks exact Schema/Scope and authorization evidence")
        sources = []
        default = False
        for ref in result["action_inputs"][2:]:
            if ref == self.fixed["task"]:
                default = True
            else:
                try:
                    if Bundle(self.root).document(ref).get("kind") == "source":
                        sources.append(ref)
                        continue
                except ProtocolError:
                    pass
                raise UnsupportedError("finite dynamic action supports only exact Schema/Scope and Source or Task default_start inputs")
        if result["next_action"] == "verify":
            if default or len(sources) != 1 or not Bundle(self.root).source(sources[0]):
                raise ProtocolError("verify requires exactly one explicit nonempty Source")
        elif result["next_action"] != "implement" or not ((len(sources) == 1 and not default) or (not sources and default)):
            raise ProtocolError("implement requires exactly one explicit Source or Task default_start")
        return {"mode": "verify_only" if result["next_action"] == "verify" else "code", "seed": sources[0] if sources else None}

    def execute(self, request):
        # Caller already owns the original native writer lease. Do not invoke
        # Backend.execute/api.implement, which would acquire a second lease.
        with self._locked(), use_authority(self.owner.path):
            self._writing_floors(request)
            from dataclasses import asdict
            native = asdict(request)
            native = jsonable(native)
            if self.state["native_request"] not in (None, native):
                raise ProtocolError("dynamic continuation requires original native settings/workspace")
            self.state["native_request"] = native
            self._save("native_bound")
            try:
                while self.state["phase"] != "done":
                    if self.state["current"] is None:
                        if self.owner.fences():
                            self.state.update(status="blocked", recovery="settle_only", reason="original owner operation unresolved")
                            self._save("owner_fenced")
                            break
                        self._ready()
                    self._publish_request()
                    current = self.state["current"]
                    context, result, campaign = self._fixed(request) if current["operation"] == "implement" else self._writing()
                    self.checkpoint("child_result_before_driver", self)
                    if result["recovery"] == "settle_only" or result["usage"]["unsettled"] or self.owner.fences():
                        self.state.update(status="blocked", recovery="settle_only", reason=result["stop_reason"])
                        self._save("original_operation_unknown")
                        break
                    p = Publication(context.root, current["publication"])
                    ref = p.put_json("dynamic-selected-result.json", result)
                    p.commit([ref], subjects=[context.request_ref])
                    incumbent = result.get("extensions", {}).get("information", {}).get("native_incumbent")
                    import_bundle(context.bundle, self.root, [ref, *([incumbent] if incumbent else [])], subjects=[context.request_ref])
                    entry = {"result": ref, "request": context.request_ref, "out": current["out"],
                             "schema": context.request.get("schema"), "native": None}
                    if campaign and context.load_state("aka-fixed-backend")["finished"]:
                        entry["native"] = close_stage(context, campaign)
                        if entry["native"]["head"] and context.request["mode"] == "code":
                            self.state["handoff"] = entry["native"]
                    elif campaign and result["status"] == "completed":
                        raise ProtocolError("completed fixed result cannot advance before its actual finish settles")
                    self.state["history"].append(entry)
                    self.state["current"] = None
                    self.state.update(status=result["status"], recovery=result["recovery"], reason=result["stop_reason"])
                    phase = self.state["phase"]
                    if result["status"] != "completed":
                        self.state["phase"] = "done"
                    elif phase == "design":
                        self.state.update(schema=result["schema"], scope=result["scope"], phase="fixed", status="partial",
                                          recovery="continue", reason="design published; fixed implementation not admitted")
                    elif phase == "optimize":
                        self.state["optimize_count"] += 1
                        self.state.update(schema=result["schema"], scope=result["scope"])
                        if result["next_action"] == "none":
                            self.state.update(phase="done", reason="optimize returned no action")
                        else:
                            try:
                                action = self._action(result)
                            except ProtocolError as error:
                                self.state.update(phase="done", status="blocked", recovery="new_request", reason=str(error))
                            else:
                                self.state.update(phase="followup", action=action, status="partial", recovery="continue",
                                                  reason="writing published; next fixed implementation not admitted")
                    elif phase == "followup" or self.config["optimize_request"] is None:
                        self.state["phase"] = "done"
                    else:
                        self.state.update(phase="optimize", status="partial", recovery="continue",
                                          reason="fixed stage settled; one authorized optimize remains")
                    self._save("writing_published" if current["operation"] != "implement" else "old_scope_closed")
            except BudgetDenied as error:
                fenced = bool(self.owner.fences())
                self.state.update(status="blocked" if fenced else "partial", recovery="settle_only" if fenced else "none", reason=str(error))
                self._save("admission_denied")
            value = self.summary()
            durable_write(self.root / "dynamic-result.json", jcs(value), replace=True)
            return value

    def summary(self):
        qualified = None
        native = None
        for entry in self.state["history"]:
            result = Bundle(self.root).document(entry["result"])
            if result["operation"] == "implement":
                native = {"schema": entry["schema"], "request": entry["request"], "result": entry["result"],
                          "incumbent": result["extensions"]["information"]["native_incumbent"],
                          "head": entry["native"]["head"] if entry["native"] else None}
                if entry["schema"] == self.state["schema"]:
                    qualified = result["code"]["qualified"]
        if self.state["phase"] in {"fixed", "followup"}:
            qualified = None  # A newly requested fixed stage owns its own result.
        return {"format": VERSION, **{k: self.state[k] for k in ("phase", "status", "recovery", "reason", "schema", "scope", "optimize_count", "history")},
                "current": self.state["current"], "current_qualified": qualified,
                "native_result": native, "owner": self.owner.descriptor, "fences": self.owner.fences()}

    def close(self):
        for child in self.children:
            child.close()
        self.native.close()


def jsonable(value):
    """Serialize frozen native DTOs without changing their values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value
