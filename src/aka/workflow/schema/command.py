"""Thin native tool/CommandEvaluator seam; full Task only, no comparison policy."""
from __future__ import annotations

from dataclasses import replace
import json

from aka.contracts.evaluation import CommandEvaluation, EvaluationRequest
from aka.contracts.execution import CommandSpec, UnsupportedExecution
from aka.evaluation.tool_requests import ToolRequest, parse_tool_arguments


def request_from_argv(argv, *, workspace, evaluation_input, configuration, workspace_provider,
                      defaults=None, filesystem_access=None):
    args = parse_tool_arguments(argv)
    if args.kind in {"record-read", "kernel-read", "kernel-records", "kernel-pareto-frontier", "env"}:
        return ToolRequest(args.kind, None, args)
    baseline = vars(parse_tool_arguments([]))
    allowed = {"kind", "version", "evaluation_mode", "no_sync"}
    changed = {key for key, value in vars(args).items() if value != baseline[key]}
    if args.kind not in {"auto", "run"} or args.evaluation_mode not in {None, "full"} or changed - allowed:
        raise UnsupportedExecution("Schema Task owns all cases/timing; subset, overrides, ad-hoc commands and native ABBA need separate support")
    # Original defaults (six cases, twenty timed calls) do not override the Task.
    command = ("kernelschema-evaluate", args.version or "")
    return ToolRequest("run", CommandSpec(workspace, command, operation="run", timeout=configuration.timeout,
        hardware=configuration.hardware, filesystem_access=filesystem_access), args)


class SchemaCommandEvaluator:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def prepare(self, command):
        if (command.operation != "run" or len(command.command) != 2
                or command.command[0] != "kernelschema-evaluate" or command.submission is not None
                or command.input_files or command.prepared_inputs or command.source_archives):
            raise UnsupportedExecution("only the host's full Schema Task command is supported")
        mode = "v0" if command.command[1] == "v0" else "schema_task"
        from .evaluation import SchemaEvaluator
        selected = SchemaEvaluator(replace(self.evaluator.configuration,
            timeout=min(command.timeout, self.evaluator.configuration.timeout)), self.evaluator.workspace,
            self.evaluator.evaluation_input, execution=self.evaluator.execution, operation=self.evaluator.operation)
        return selected.prepare(EvaluationRequest(command.workspace, mode=mode, cancel_event=command.cancel_event))

    def observe_step(self, prepared, index, result):
        return self.evaluator.observe_step(prepared, index, result)

    def fallback(self, prepared, results):
        return ()

    def interpret(self, prepared, results):
        verification = self.evaluator.interpret(prepared, results)
        # Only public status and coverage-shape aggregate performance are projected.
        # Private case/seed/guard values and raw worker output stay in KS/Experiment.
        payload = dict(all_pass=verification.passed, correct=True if verification.passed else
                       False if verification.gate == "FAIL" else None,
                       latency_us=verification.candidate_latency_us,
                       latency_us_geomean=verification.candidate_latency_us,
                       latency_us_by_shape=dict(verification.candidate_latency_us_by_shape),
                       measurement_scope="full-task", comparison=None,
                       error=None if verification.passed else verification.failure.message)
        if verification.failure is not None:
            from dataclasses import asdict
            payload["evaluation_failure"] = asdict(verification.failure)
        result = replace(results[0], stdout="[test_kernel] RESULT_JSON=" + json.dumps(payload, allow_nan=False) + "\n",
                         stderr="", exit_code=0 if verification.passed else 1)
        return CommandEvaluation(result, verification=verification, artifacts=verification.artifacts)
