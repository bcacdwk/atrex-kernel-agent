"""Optional accounting over normal AKA calls using an existing KS account.

No evaluator, model transport or account is created here. A finite group allows
the native Agent to call normal GPU tools while its model invocation is active.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import json
import threading
import time


class StandardBudget:
    def __init__(self, path, *, workspace, identity, limits, run_id=None):
        from kernelschema.budget import OwnerAccount
        from .standard import _digest
        if (not isinstance(limits, dict) or set(limits) != {"model_calls", "evaluator_jobs"}
                or any(type(n) is not int or n <= 0 for n in limits.values())):
            raise ValueError("owner_limits must explicitly bound model_calls and evaluator_jobs")
        self.account = OwnerAccount(path)
        if self.account.descriptor["limits"]["tokens"] is not None:
            raise ValueError("native AKA reports tokens after an invocation; this adapter requires an unbounded parent token field")
        self.workspace = Path(workspace)
        self.identity = identity
        self.group_id = "aka-standard-" + _digest([str(self.workspace), identity])[7:]
        self.run_id = run_id or self.group_id
        self.limits, self.parent_run = limits, run_id is not None
        self.directory = self.workspace.parent / ".aka-schema-budget" / self.group_id
        self.reservations = None
        self.lock = threading.RLock()
        self.jobs = {}

    def _prepare(self):
        from kernelschema.budget import cost, BudgetDenied
        from .standard import _digest
        if self.reservations is not None:
            return
        if not self.parent_run:
            self.account.register(self.run_id, _digest([self.identity, self.limits]),
                self.account.descriptor["limits"], root=str(self.workspace))
        operations = []
        for role, count in (("coder", self.limits["model_calls"]),
                            ("evaluator", self.limits["evaluator_jobs"])):
            for index in range(count):
                operations.append({"operation_id": f"{self.group_id}-{role}-{index}",
                    "role": role, "purpose": "aka-standard-" + role,
                    "amount": cost(role=role, calls=int(role == "coder"),
                                   evaluator_jobs=int(role == "evaluator"))})
        self.reservations = self.account.reserve_group(self.run_id, self.group_id, operations)
        prior = self.account.operations(self.run_id)
        if any(prior[key]["launched"] or prior[key]["released"] for key in self.reservations):
            self.reservations = None
            raise BudgetDenied("the original native run already started; reconcile its retained workspace and receipts")
        self.directory.mkdir(parents=True, exist_ok=True)

    def take(self, role):
        from kernelschema.budget import BudgetDenied
        with self.lock:
            self._prepare()
            operations = self.account.operations(self.run_id)
            for key, reservation in self.reservations.items():
                operation = operations[key]
                if operation["role"] == role and not operation["launched"] and not operation["released"]:
                    if not self.account.authorize(reservation):
                        raise BudgetDenied("original account denied this native operation")
                    return reservation
            raise BudgetDenied("native " + role + " allocation exhausted")

    def _record(self, reservation, *, role, facts, calls, jobs, tokens, terminated):
        from kernelschema.artifacts import durable_write
        key = reservation["operation_id"]
        old = self.account.operations(self.run_id)[key]["history"]
        sequence = len(old)
        row = {"producer": "aka-standard", "operation_id": key, "role": role,
               "purpose": "aka-standard-" + role, "receipt_id": f"{key}:{sequence}",
               "sequence": sequence, "receipt": None, "state": "settled" if terminated else "unknown",
               "calls": calls, "evaluator_jobs": jobs, "tokens": tokens, "gpu_ms": None}
        durable_write(self.directory / f"{key}-{sequence}.json",
                      json.dumps({"usage": row, "native": facts}, ensure_ascii=False).encode())
        self.account.absorb(self.run_id, row, terminated=terminated,
                            lower_bounds={"calls": calls or 0, "evaluator_jobs": jobs or 0})
        if not terminated:
            self.account.mark_unknown(key, "native execution has no confirmed terminal result")

    def model(self, invoke, *, cancel=None):
        reservation = self.take("coder")
        timer = None
        if cancel is not None:
            remaining = max(0.001, reservation["deadline_utc_ms"] / 1000 - time.time())
            timer = threading.Timer(remaining, cancel)
            timer.daemon = True
            timer.start()
        try:
            result = invoke(reservation)
        except BaseException:
            self.account.mark_unknown(reservation["operation_id"], "native model invocation did not return its receipt")
            raise
        finally:
            if timer is not None:
                timer.cancel()
        self._record(reservation, role="coder", facts={
            "session_id": result.session_id, "exit_status": result.exit_status,
            "timed_out": result.timed_out, "usage": asdict(result.terminal_usage),
            "observation_errors": list(result.observation_errors)},
            calls=1, jobs=0, tokens=result.terminal_usage.total_tokens, terminated=True)
        return result

    def execution(self, reservation, result):
        with self.lock:
            ids = {ref.identifier for ref in result.job_references}
            ids.update(attempt.resource.identifier for attempt in result.attempts
                       if attempt.resource and attempt.resource.identifier)
            if result.resource and result.resource.identifier:
                ids.add(result.resource.identifier)
            # Re-observing a retained job is not another GPU submission.
            counted = {identifier for identifier in ids if identifier not in self.jobs
                       or self.jobs[identifier] == reservation}
            for identifier in counted:
                self.jobs[identifier] = reservation
            terminated = result.facts.disposition in {"terminal", "not_submitted"}
            count = len(counted) if ids else (0 if result.facts.disposition == "not_submitted" else None)
            self._record(reservation, role="evaluator", facts=result.as_dict(), calls=0,
                         jobs=count, tokens=0, terminated=terminated)

    def close(self):
        with self.lock:
            if self.reservations is None:
                return
            operations = self.account.operations(self.run_id)
            for key in self.reservations:
                if not operations[key]["launched"] and not operations[key]["released"]:
                    self.account.release_unstarted(key)


class BudgetExecutor:
    """Keep original request preparation, submission and result interpretation."""
    def __init__(self, native, budget):
        self.native, self.budget = native, budget

    def __getattr__(self, name):
        return getattr(self.native, name)

    def bind(self, context):
        return type(self)(self.native.bind(context), self.budget)

    @property
    def recovery_capabilities(self):
        # Reading/cancelling a known job does not consume a new submission slot.
        # Unknown submissions remain with their original receipt for settlement.
        return replace(self.native.recovery_capabilities, idempotent_replay=False)

    def replay_job(self, command, *, idempotency_key):
        from kernelschema.budget import BudgetDenied
        raise BudgetDenied("settle the original native job before another submission")

    def query_job(self, reference, **kwargs):
        result = self.native.query_job(reference, **kwargs)
        reservation = self.budget.jobs.get(reference.identifier)
        if reservation is not None:
            self.budget.execution(reservation, result)
        return result

    def cancel_job(self, reference, **kwargs):
        result = self.native.cancel_job(reference, **kwargs)
        reservation = self.budget.jobs.get(reference.identifier)
        if reservation is not None:
            self.budget.execution(reservation, result)
        return result

    def run(self, command):
        reservation = self.budget.take("evaluator")
        # Original cancellation handles remote identity; a timeout never implies
        # remote termination. Parent expiry therefore retains unknown job fences.
        cancel = threading.Event()
        original = command.cancel_event
        class Cancellation:
            def is_set(self):
                return cancel.is_set() or (original is not None and original.is_set())
        remaining = max(0.001, reservation["deadline_utc_ms"] / 1000 - time.time())
        timer = threading.Timer(remaining, cancel.set)
        timer.daemon = True
        timer.start()
        try:
            result = self.native.run(replace(command, cancel_event=Cancellation()))
        except BaseException as error:
            result = getattr(error, "execution_outcome", None)
            if result is not None:
                self.budget.execution(reservation, result)
            else:
                self.budget.account.mark_unknown(reservation["operation_id"], "native executor did not return its receipt")
            raise
        finally:
            timer.cancel()
        self.budget.execution(reservation, result)
        return result
