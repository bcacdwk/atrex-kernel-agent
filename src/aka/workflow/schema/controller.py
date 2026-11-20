"""Fixed R transactions using the original native Campaign and episode engine."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from aka.contracts.evaluation import EvaluationRequest
from aka.contracts.workspace import CandidateHandle
from aka.experiment.runner import LegacyExperiment
from aka.workflow.light.long_horizon.campaign import LongHorizonCampaign
from aka.workflow.light.long_horizon.models import EpisodeHandoff, SessionResult, InvocationObservation
from aka.workflow.light.long_horizon.recorded_verifier import RecordedABBAValidator
from aka.workflow.light.long_horizon.store import CampaignStore
from kernelschema import fidelity
from kernelschema.artifacts import Publication, import_bundle, durable_write, read_regular, install_source
from kernelschema.budget import BudgetDenied, cost
from kernelschema.contracts import PROTOCOL, ProtocolError, UnsupportedError, digest, jcs, loads
from kernelschema.evaluation import RowSoftmaxTask
from kernelschema.runtime import RoleDelivery, AdmissionDenied

from .calls import RoleCalls, Unsettled
from .comparison import ComparisonCallback
from .evaluation import EvaluationOperation, SchemaEvaluatorFactory, native_verification
from .preparation import SeedLifecycle, NAMESPACE as SOURCE_NAMESPACE
from .projection import assessment, apply_report, qualifies, result as public_result
from .source import publish_native_source
from .state import Transactions


class FixedController:
    def __init__(self, run, campaign, execution, services, *, continuing=False, checkpoint=None, selection=None):
        self.run, self.campaign, self.execution, self.services = run, campaign, execution, services
        c = campaign
        if c.framework_baseline != "never" or c.convert_after or c.framework.lower() != "triton":
            raise UnsupportedError("fixed AKA requires Triton, framework_baseline=never and convert_after=0")
        if c.module_options.search or c.module_options.goal_episodes:
            raise UnsupportedError("fixed AKA cannot run search/planner/goal episodes")
        if c.token_budget or run.request["budget"]["limits"]["tokens"] is not None:
            raise UnsupportedError("native no-tools transport cannot enforce finite token budgets")
        native_limits = run.objects["profile"]["native_limits"].get("workflow_limits")
        if native_limits is not None and native_limits.get("tokens") is not None:
            raise UnsupportedError("native no-tools transport cannot enforce finite native token budgets")
        self.task = RowSoftmaxTask(run.bundle, run.request["task"])
        self.task.require_evaluation_scope(run.request["scope"])
        self.coverage_ids = self.task.evaluation_capabilities()["coverage_ids"]
        self.scope = self.task.validate_scope(run.request["schema"], run.request["scope"])
        binding = {"workspace": str(c.workspace.absolute()), "selection": selection,
            "task": run.request["task"], "schema": run.request["schema"], "mode": run.request["mode"],
            "native": {k: getattr(c, k) for k in ("max_iters", "token_budget", "framework", "platform", "arch",
                "framework_baseline", "convert_after", "optimization_mode", "agent_cli", "verify_repeats",
                "verify_run_timeout", "min_improvement_pct")},
            "services": services.identity(), "source_format": c.module_options.file_system}
        binding["native"]["min_improvement_pct"] = str(c.min_improvement_pct)
        self.tx = Transactions(run, binding, continuing=continuing, checkpoint=checkpoint)
        if self.tx.owner.descriptor["limits"]["tokens"] is not None:
            raise UnsupportedError("native no-tools transport cannot enforce finite parent tokens")
        self.calls = RoleCalls(self.tx, c.agent)
        self.sources = None
        execution.authorize = self.authorize_evaluation
        execution.launched = lambda identifier: self.tx.owner.operations(run.run_id)[identifier]["launched"]
        execution.settled = self.absorb_evaluation
        execution.active = lambda identifier: self.tx.owner.check_active(self.owned(identifier)["reservation"])
        self.evaluator_factory = SchemaEvaluatorFactory(execution=execution, operation=self.resolve_evaluation)
        self.comparison = ComparisonCallback(self, getattr(services, "comparison", None))
        verifier = RecordedABBAValidator(execute=self.comparison, workspace_provider=c.candidate_workspace,
            hardware=c.platform, repeats=c.verify_repeats, per_run_timeout=c.verify_run_timeout,
            min_improvement_pct=c.min_improvement_pct)
        self.engine = LongHorizonCampaign(c, max_version=c.max_iters, token_budget=c.token_budget,
            handoff_resumes=0, max_stall=c.max_stall, verifier=verifier,
            session_runner=SimpleNamespace(run=self.run_session), candidate_audit=self)
        self._original_finish = c._finish
        self._original_v0 = c._run_v0_evaluator_with_correctness_prefetch
        c._run_v0_evaluator_with_correctness_prefetch = self.v0_with_prefetch
        c._ensure_framework_baseline_correctness_guidance = self.prefetch
        c._review_production_candidate = self.production_review
        c._evaluator_directive = lambda: "The fixed Schema Task owns complete evaluation; the host invokes the evaluator."
        # Any legacy helper unexpectedly reaching the unrestricted session path
        # is denied. Every supported helper goes through the finite RoleCalls.
        c._run_session = self.unrouted_session

    def unrouted_session(self, *args, **kwargs):
        raise UnsupportedError("legacy helper has no fixed-role owner admission")

    def active_episode(self):
        _, active = CampaignStore.checkpoint(self.campaign.workspace)
        return active["episode"] if active else None

    def owned(self, identifier):
        matches = [op for op in self.tx.state["operations"].values() if op["id"] == identifier]
        if len(matches) != 1:
            raise ProtocolError("unknown original AKA operation")
        return matches[0]

    def authorize_evaluation(self, identifier):
        op = self.owned(identifier)
        if not any(i["operation_id"] == identifier for i in self.run.intents()):
            raise ProtocolError("evaluation launch preceded durable owner intent")
        return self.tx.owner.authorize(op["reservation"])

    def absorb_evaluation(self, identifier, terminal, bundle):
        op = self.owned(identifier)
        references = [terminal["usage_row"]["receipt"]]
        if terminal.get("report"):
            references.append(terminal["report"])
        envelope = loads(read_regular(self.execution.root / identifier, "intent.json")[0])
        import_bundle(bundle, self.run.root, references,
                      subjects=[self.run.request["task"], envelope["source_binding"]["source"]])
        self.tx.absorb(op, terminal["usage_row"], evidence=references)
        op["report"] = terminal.get("report")
        self.tx.save("evaluation_absorbed")

    def resolve_evaluation(self, request):
        episode = self.active_episode()
        v0 = request.mode == "v0" or (self.run.request["mode"] == "code" and episode is None)
        if v0:
            source = self.run.objects["task"]["baseline_source"]
            key, episode = "v0", None
        else:
            source = publish_native_source(self.run.root, request.workspace,
                self.campaign.candidate_workspace, revision=request.candidate_commit or None).source
            key = f"e{episode}-evaluation" if episode else "verify-only-evaluation"
        op = self.tx.plan(key, role="evaluator", source=source, episode=episode)
        self.tx.reserve(op, floor=self.tx.finish_floor(evaluate=False,
            review=4 if self.campaign.optimization_mode == "production" else 1))
        self.tx.intent(op, self.execution.locator(op["id"]))
        return EvaluationOperation(op["id"], op["id"], self.run.owner["id"], op["reservation"],
                                   expected_source=source, mode="v0" if v0 else "schema_task")

    def evaluate(self, workspace, source, *, episode=None):
        key = f"e{episode}-evaluation" if episode else "verify-only-evaluation"
        op = self.tx.state["operations"].get(key)
        if op and op.get("report"):
            evidence = self.execution.read_operation(op["id"])
        else:
            from aka.contracts.evaluation import EvaluationConfiguration
            evaluator = self.evaluator_factory.create(EvaluationConfiguration(timeout=min(1200, self.campaign.sandbox_timeout)),
                candidate_workspace=self.campaign.candidate_workspace, evaluation_input=self.campaign.problem.evaluation)
            try:
                LegacyExperiment(self.execution, evaluator).run(EvaluationRequest(Path(workspace), mode="schema_task"))
            except (OSError, ProtocolError, RuntimeError):
                op = self.tx.state["operations"].get(key)
                if op and self.tx.owner.operations(self.run.run_id).get(op["id"], {}).get("launched"):
                    try:
                        evidence = self.execution.read_operation(op["id"])
                    except (OSError, ProtocolError):
                        self.tx.unknown(op, "original Task evaluation remains unresolved")
                        raise Unsettled(self.tx.state["stop_reason"])
                else:
                    raise
            op = self.tx.state["operations"][key]
            evidence = self.execution.read_operation(op["id"])
        if evidence["source_binding"]["source"]["sha256"] != source["sha256"]:
            raise ProtocolError("full Task result differs from exact current C")
        if evidence["report"]:
            apply_report(self.tx.state, source, evidence["report_ref"], evidence["report"], ("build", "correctness", "performance"))
        self.tx.save("task_assessed")
        return native_verification(evidence)

    def review(self, source, *, episode=None):
        key = f"e{episode}-fidelity" if episode else "verify-only-fidelity"
        previous = self.tx.state["operations"].get(key)
        if previous and previous.get("fidelity"):
            reference = previous["fidelity"]
            report = self.run.bundle.document(reference, "fidelity")
        else:
            # The same method/inventory/input Ref is retained before reviewer fees.
            ref_key = key + "-input"
            if ref_key not in self.tx.state:
                publication = Publication(self.run.root)
                reference = publication.put_bytes("fidelity-input.json", jcs(fidelity.review_input(self.run, source, self.scope)))
                publication.commit([reference])
                self.tx.state[ref_key] = reference
                self.tx.save("fidelity_input_ready")
            delivery = RoleDelivery("finite_obligations", self.tx.state[ref_key], ("reviewer",))
            _, _, op, receipt = self.calls.invoke(key, "reviewer", source, episode=episode, deliveries=(delivery,),
                floor=self.tx.finish_floor(evaluate=False, review=3 if self.campaign.optimization_mode == "production" else 0))
            if "fidelity_publication" not in op:
                op["fidelity_publication"] = uuid4().hex
                self.tx.save("fidelity_publication_intent")
            reference, report = fidelity.report(self.run, source=source, operation_id=op["id"],
                scope_validation=self.scope, receipt=receipt, publication=Publication(self.run.root, op["fidelity_publication"]))
            op["fidelity"] = reference
        value = apply_report(self.tx.state, source, reference, report, ("coverage", "fidelity"))
        self.tx.save("fidelity_assessed")
        return value

    def v0_with_prefetch(self):
        labels = tuple(self.campaign._framework_baseline_correctness_reviewers())
        if labels:
            source = self.run.objects["task"]["baseline_source"]
            ops = [self.tx.plan("v0", role="evaluator", source=source)]
            ops.extend(self.tx.plan("v0-prefetch-" + label, role="reviewer", source=source) for label in labels)
            owned_before = self.tx.owner.operations(self.run.run_id)
            recovering_group = any(op["id"] in owned_before for op in ops)
            minimum = self.run.objects["profile"]["parameters"].get("minimum_active_ms",
                {"coder": 1000, "evaluator": 1000, "reviewer": 1000})
            rows = [{"operation_id": op["id"], "role": op["role"], "purpose": op["purpose"],
                     "amount": cost(role=op["role"], calls=int(op["role"] == "reviewer"),
                                    evaluator_jobs=int(op["role"] == "evaluator"), active_ms=minimum[op["role"]])} for op in ops]
            group = self.tx.owner.reserve_group(self.run.run_id, "aka-v0-" + self.run.run_id, rows,
                floor=self.tx.finish_floor(), native_limits=self.run.objects["profile"]["native_limits"].get("workflow_limits"))
            for op in ops:
                op["reservation"] = group[op["id"]]
            self.tx.save("v0_group_reserved")
            owned = self.tx.owner.operations(self.run.run_id)
            if recovering_group and not any(owned[op["id"]]["launched"] for op in ops):
                self.tx.owner.claim_group("aka-v0-" + self.run.run_id, group)
        return self._original_v0()

    def prefetch(self):
        source = self.run.objects["task"]["baseline_source"]
        for label in self.campaign._framework_baseline_correctness_reviewers():
            key = "v0-prefetch-" + label
            try:
                self.calls.invoke(key, "reviewer", source)
            except BaseException as error:
                op = self.tx.state["operations"][key]
                owned = self.tx.owner.operations(self.run.run_id)[op["id"]]
                if not owned["launched"]:
                    self.tx.owner.release_unstarted(op["id"])
                self.tx.state["diagnostics"].append("V0 prefetch " + label + ": " + str(error))
                self.tx.save("prefetch_interrupted")
                raise

    def production_review(self, workspace, framework, require_gluon):
        from aka.workflow.light.orchestration.campaign import DEPENDENCY_REVIEW_PROMPT, _validate_production_review
        from aka.workflow.light.orchestration.session_io import _production_review_digest, _production_review_candidate_paths
        source = publish_native_source(self.run.root, workspace, self.campaign.candidate_workspace).source
        native_digest = _production_review_digest(workspace, framework, require_gluon,
            workspace_provider=self.campaign.candidate_workspace)
        paths = _production_review_candidate_paths(workspace, workspace_provider=self.campaign.candidate_workspace)
        files = frozenset(p.relative_to(workspace).as_posix() for p in paths)
        key = "production-" + native_digest
        op = self.tx.state["operations"].get(key)
        if op:
            if op["source"]["sha256"] != source["sha256"]:
                raise ProtocolError("original production review source changed")
            source = op["source"]  # Preserve its original published Ref on recovery.
        if op and "production_errors" in op:
            return op["production_errors"]
        if key + "-input" not in self.tx.state:
            publication = Publication(self.run.root)
            packet = {"framework": framework, "required_phase": "gluon" if require_gluon else "selected_framework",
                "candidate_digest": native_digest, "candidate_files": sorted(files),
                "metadata": {p.relative_to(workspace).as_posix(): read_regular(p.parent, p.name)[0].decode()
                             for p in paths if p.name != "kernel.py"}}
            ref = publication.put_bytes("production-policy.txt", DEPENDENCY_REVIEW_PROMPT.read_bytes() +
                b"\nHost supplies kernel.py as current_source and metadata below. No filesystem tools exist. Return the requested dependency_review.json object as RoleOutput review content.\n" + jcs(packet))
            publication.commit([ref])
            self.tx.state[key + "-input"] = ref
            self.tx.save("production_input_ready")
        delivery = self.run.root / ".runtime/aka-delivery"
        floor_reviews = 2 if self.active_episode() is not None else 1 if Path(workspace) == delivery else 0
        output, _, op, _ = self.calls.invoke(key, "reviewer", source,
            deliveries=(RoleDelivery("native_production_policy", self.tx.state[key + "-input"], ("reviewer",)),),
            floor=self.tx.finish_floor(evaluate=False, review=floor_reviews))
        errors = ["independent native production review unavailable"]
        if output.output:
            errors, _ = _validate_production_review(output.output.content, candidate_files=files)
        if _production_review_digest(workspace, framework, require_gluon,
                workspace_provider=self.campaign.candidate_workspace) != native_digest:
            raise ProtocolError("native production source/metadata changed during review")
        op["production_errors"] = errors
        self.tx.save("production_reviewed")
        return errors

    def admit(self, *, phase, episode, workspace, active):
        self.tx.validate()
        # Original pending operations are reconciled by their own evaluator/role
        # transactions. Never grant a new episode while another operation is unknown.
        if self.tx.owner.fences() and active is None and phase == "episode":
            return False
        return True

    def settled(self, *, episode, workspace, attempt):
        entry = self.tx.state["episodes"].setdefault(str(episode), {})
        retained = {"native_json": json.dumps(attempt, sort_keys=True), "accepted": attempt.get("accepted")}
        if entry.get("attempt") and entry["attempt"] != retained:
            raise ProtocolError("native attempt changed after settlement")
        entry["attempt"] = retained
        self.tx.state["latest"] = self.sources.current_source(episode).source
        self.tx.save("episode_settled")

    def run_session(self, workspace, prompt, *, output_directory, **unused):
        episode = self.active_episode()
        entry = self.tx.state["episodes"].setdefault(str(episode), {})
        original = self.tx.state["operations"].get(f"e{episode}-coder")
        source = original["source"] if original else self.sources.current_source(episode).source
        deliveries = tuple(RoleDelivery("feedback", r, ("coder",)) for r in self.tx.state["feedback"][-1:])
        output, authority, op, _ = self.calls.invoke(f"e{episode}-coder", "coder", source, episode=episode,
            deliveries=deliveries, floor=self.tx.finish_floor(review=4 if self.campaign.optimization_mode == "production" else 1))
        invocation = output.invocation
        observation = InvocationObservation(invocation.terminal_usage, invocation.events, invocation.capabilities,
            invocation.observation_errors, invocation.resume_usage_qualified, invocation.session_id)
        if output.output is None:
            return SessionResult(1, invocation.timed_out, invocation.terminal_usage.total_tokens or 0,
                invocation.session_id, 0, None, invocations=(observation,),
                completion_diagnosis=", ".join(output.permission_errors))
        binding = self.sources.install_result(episode, output, authority=authority)
        self.tx.state["latest"] = binding.source
        self.evaluate(workspace, binding.source, episode=episode)
        self.review(binding.source, episode=episode)
        active = CampaignStore(self.campaign.workspace, candidate_workspace=self.campaign.candidate_workspace).load_active()
        handle = CandidateHandle(episode, active["base_commit"], active["episode_branch"], Path(active["worktree"]))
        runtime = self.campaign._supervisor_runtime
        registered = getattr(runtime, "journals", {}).get(Path(workspace))
        view = registered.view if registered else getattr(runtime, "views", {}).get(Path(workspace))
        if view is None:
            raise ProtocolError("original native EpisodeWorkspace is unavailable")
        if not entry.get("candidate_commit"):
            view.publish()
            commit = self.campaign.candidate_workspace.seal_candidate(handle, output.native_source())
            entry["candidate_commit"] = commit
            self.tx.save("candidate_sealed")
        else:
            commit = entry["candidate_commit"]
        handoff = EpisodeHandoff("candidate_ready", commit)
        self.qualify(handle, commit, active)
        journal_path = handle.path / ".atrex_long_horizon/journal.json"
        journal = loads(read_regular(journal_path.parent, journal_path.name)[0])
        journal.update(state="candidate_ready", candidate_commit=commit,
            candidate_kernel_digest="sha256:" + binding.native_identity,
            finalized_at=datetime.now(timezone.utc).isoformat(),
            outcome={"summary": "Fixed Schema host completed exact-source evaluation and independent review", "next_directions": []},
            schema_qualification=entry["qualification"])
        durable_write(journal_path, jcs(journal), replace=True)
        durable_write(handle.path / ".atrex_long_horizon/handoff.json", jcs(handoff.as_dict()), replace=True)
        self.tx.save("native_handoff_ready")
        return SessionResult(invocation.exit_status, invocation.timed_out, invocation.terminal_usage.total_tokens or 0,
            invocation.session_id, 0, handoff, invocations=(observation,))

    def completion_check(self, worktree, journal_path, handoff):
        entry = self.tx.state["episodes"].get(str(worktree.episode), {})
        if handoff.status != "candidate_ready" or entry.get("candidate_commit") != handoff.candidate_commit:
            return "fixed host has no matching sealed terminal"
        return self.campaign.candidate_workspace.validate_candidate(worktree, handoff.candidate_commit)[0]

    def qualify(self, worktree, candidate_commit, active):
        entry = self.tx.state["episodes"].setdefault(str(worktree.episode), {})
        source = self.sources.current_source(worktree.episode).source
        sealed = publish_native_source(self.run.root, worktree.path, self.campaign.candidate_workspace,
                                      revision=candidate_commit, expected=source)
        if entry.get("qualification"):
            old = entry["qualification"]
            if old["source"]["sha256"] != source["sha256"] or old["candidate_commit"] != candidate_commit:
                raise ProtocolError("recovery cannot replace original seal qualification")
            if active.get("schema_qualification") not in (None, old):
                raise ProtocolError("native active qualification differs from original audit")
            return old
        value = assessment(self.tx.state, source)
        audit = {"format": "aka.schema.qualification.v1", "run_id": self.run.run_id,
            "owner": self.run.owner, "request_sha256": self.run.request_sha256,
            "schema": self.run.request["schema"], "scope": self.run.request["scope"], "rules": self.run.request["rules"],
            "source": source, "native_identity": sealed.native_identity, "candidate_commit": candidate_commit,
            "base_commit": worktree.base_commit, "episode": worktree.episode, "qualified": qualifies(value),
            "assessment": loads(jcs(value)), "operations": [op["terminal"] for op in self.tx.state["operations"].values()
                if op["episode"] == worktree.episode and op["terminal"]]}
        entry["qualification"] = audit
        self.tx.save("qualification_durable")
        return audit

    def attempt(self, active, worktree):
        entry = self.tx.state["episodes"].get(str(worktree.episode), {})
        return entry.get("qualification", {"qualified": False, "reason": "no complete candidate qualification"})

    def usage(self, active, worktree):
        return [op["terminal"] for op in self.tx.state["operations"].values()
                if op["episode"] == worktree.episode and op["terminal"]]

    def recover_promotion(self, active, commit, evidence):
        audit = evidence.get("schema_qualification") if evidence else None
        if (not audit or audit.get("run_id") != self.run.run_id or audit.get("owner") != self.run.owner
                or audit.get("request_sha256") != self.run.request_sha256 or audit.get("qualified") is not True):
            self.tx.state.update(status="blocked", recovery="new_request", stop_reason="committed native promotion lacks its original Schema qualification")
            self.tx.state["phase"] = "audit_unknown"
            self.tx.save("promotion_qualification_unknown")
            return {"qualified": None, "reason": "native promotion retained; original Schema audit unknown"}
        actual = publish_native_source(self.run.root, self.campaign.workspace,
            self.campaign.candidate_workspace, revision=commit, expected=audit["source"])
        if actual.native_identity != audit["native_identity"]:
            raise ProtocolError("committed promotion source differs from its original Schema audit")
        for terminal in evidence.get("schema_operations", audit["operations"]):
            self.tx.owner.absorb(self.run.run_id, terminal["usage"], terminated=terminal["terminated"])
        entry = self.tx.state["episodes"].setdefault(str(active["episode"]), {})
        if entry.get("qualification") not in (None, audit):
            raise ProtocolError("committed Schema audit differs from retained qualification")
        entry["qualification"] = audit
        self.tx.save("promotion_qualification_recovered")
        return audit

    def execute(self):
        state = self.tx.state
        if state["phase"] == "done":
            return self.publish()
        if self.scope["status"] != "pass":
            state.update(status="blocked", recovery="new_request", stop_reason="fixed Schema scope is incomplete")
            return self.publish()
        try:
            fidelity.method_ref(self.run)
            admitted = self.services.admission(self.run)
        except (ProtocolError, UnsupportedError) as error:
            admitted = {"status": "unknown", "reason": str(error)}
        if admitted["status"] != "pass":
            state.update(status="blocked", recovery="new_request", stop_reason=admitted["reason"])
            return self.publish()
        try:
            self.tx.owner.begin(self.run.run_id)
            if state["phase"] == "finishing":
                pass  # Recover the original paid finish before publishing.
            elif self.run.request["mode"] == "verify_only":
                self.verify_only()
            else:
                exists = (self.run.root / f".runtime/state/{SOURCE_NAMESPACE}.json").exists()
                self.sources = (SeedLifecycle.reopen if exists else SeedLifecycle.create)(self.run, self.engine, self)
                while state["phase"] == "running":
                    checkpoint, active = CampaignStore.checkpoint(self.campaign.workspace)
                    last = state["episodes"].get(str(checkpoint.episodes), {})
                    if active is None and last.get("attempt") and qualifies(assessment(state, state["latest"])):
                        current = assessment(state, state["latest"])
                        state.update(qualified=state["latest"], status="completed", recovery="none", phase="finishing",
                            stop_reason="current exact C qualified; soft performance goal " + current["performance_goal"])
                        break
                    episode = (active["episode"] if active else checkpoint.episodes
                        if checkpoint.episodes and not last.get("attempt") else checkpoint.episodes + 1)
                    outcome = self.sources.run_episode(episode)
                    if state["phase"] == "audit_unknown":
                        break
                    if outcome.attempt is None:
                        state.update(status="partial", recovery="none", stop_reason=outcome.reason, phase="finishing")
                        break
                    current = assessment(state, state["latest"])
                    if qualifies(current):
                        state.update(qualified=state["latest"], status="completed", recovery="none", phase="finishing",
                            stop_reason="current exact C qualified; soft performance goal " + current["performance_goal"])
                    else:
                        self.feedback(current)
            if state["phase"] == "finishing":
                outcome = state.setdefault("finish_outcome", {key: state[key] for key in
                    ("status", "recovery", "stop_reason", "qualified")})
                state.update(outcome)
                self.tx.save("finishing")
                self.finish()
                state["phase"] = "done"
                self.tx.save("done")
        except (BudgetDenied, AdmissionDenied, PermissionError) as error:
            for op in state["operations"].values():
                if op.get("reservation"):
                    owned = self.tx.owner.operations(self.run.run_id)[op["id"]]
                    if not owned["launched"] and not owned["released"]:
                        self.tx.owner.release_unstarted(op["id"])
            state.update(status="partial", recovery="none", stop_reason=str(error), phase="done")
            if self.campaign.optimization_mode == "production" and not state["finished"]:
                state["qualified"] = None
        except Unsettled as error:
            state.update(status="blocked", recovery="settle_only", stop_reason=str(error))
        finally:
            self.tx.owner.end(self.run.run_id)
        return self.publish()

    def verify_only(self):
        source = self.run.request["seed"]
        if source is None or not self.run.bundle.source(source):
            raise ProtocolError("verify_only requires an explicit nonempty seed")
        root = self.run.root / ".runtime/aka-verify-only"
        if root.exists():
            if read_regular(root, "kernel.py")[0] != self.run.bundle.source(source):
                raise ProtocolError("verify_only source changed")
        else:
            install_source(self.run.bundle, source, root)
        self.evaluate(root, source)
        value = self.review(source)
        self.tx.state.update(latest=source, qualified=source if qualifies(value) else None,
            status="completed" if qualifies(value) else "partial", recovery="none", phase="finishing",
            stop_reason="verified exact seed; soft performance goal " + value["performance_goal"])

    def feedback(self, current):
        publication = Publication(self.run.root)
        ref = publication.put_json("feedback.json", {"protocol": PROTOCOL, "kind": "feedback",
            "subject": {"request_sha256": self.run.request_sha256, "task_sha256": self.run.request["task"]["sha256"],
                        "schema_sha256": self.run.request["schema"]["sha256"], "source_sha256": current["source_sha256"]},
            "reason": "Continue the same fixed Schema from exact latest C; required checks failed or unknown",
            "evidence": list({v["receipt"]["sha256"]: v["receipt"] for k, v in current.items()
                              if k in ("build", "correctness", "performance", "coverage", "fidelity") and v["receipt"]}.values())})
        publication.commit([ref], subjects=[self.run.request_ref, self.tx.state["latest"]])
        self.tx.state["feedback"].append(ref)
        self.tx.save("same_schema_feedback")

    def finish(self):
        if self.tx.state["finished"]:
            return
        if self.campaign.optimization_mode == "production":
            source = self.tx.state["latest"]
            root = self.run.root / ".runtime/aka-delivery"
            if not root.exists():
                install_source(self.run.bundle, source, root)
            elif read_regular(root, "kernel.py")[0] != self.run.bundle.source(source):
                raise ProtocolError("original delivery C changed")
            violations = self.campaign._production_kernel_violations(root)
            if violations:
                self.tx.state.update(status="partial", qualified=None,
                                     stop_reason="production C: " + "; ".join(violations))
        # No native canonical repository exists in verify_only; it never creates
        # an episode or promotes. Native _finish applies only to its actual G.
        if self.run.request["mode"] == "code":
            try:
                self._original_finish(self.tx.state["stop_reason"])
            except RuntimeError as error:
                self.tx.state["diagnostics"].append("native G export: " + str(error))
        if self.tx.owner.fences():
            raise Unsettled("original finish operation remains unresolved")
        self.tx.state["finished"] = True
        self.tx.save("finished")

    def publish(self):
        incumbent = None
        if self.campaign.candidate_workspace.current_revision(self.campaign.workspace):
            incumbent = publish_native_source(self.run.root, self.campaign.workspace,
                self.campaign.candidate_workspace,
                revision=self.campaign.candidate_workspace.current_revision(self.campaign.workspace)).source
        value = public_result(self.tx, native_incumbent=incumbent)
        self.run.publish_result(value)
        return value
