"""Schema consumes original native evidence; it does not verify or run kernels."""
from __future__ import annotations

import json
import math

from aka.evaluation.measurement import gateway_identity
from aka.evaluation.private_inputs import selected_content
from aka.evaluation.shapes import shape_batch_request
from aka.workflow.light.measurements import RecordedMeasurements


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _full_task(records, record, task, hardware):
    """Bind each prepared input to this Task, including batched references.

    A prepared schedule only proves its own requested Shapes. Read the original
    owner-checked input artifacts as well, rather than upgrading workload_scope
    or trusting a matching set of Shape names from a different reference.
    """
    content = selected_content(task)
    fields = {"reference.py": "reference_py", "input.py": "input_py",
              "shapes.json": "shapes", "metadata.json": "metadata", "roofline.json": "roofline"}
    if not {"reference.py", "input.py", "shapes.json", "metadata.json"} <= content.keys():
        return False
    reference = {field: json.loads(content[name].content) if name.endswith(".json")
                 else content[name].content.decode() for name, field in fields.items() if name in content}
    shapes = set(task.case_ids or reference["shapes"])
    if not shapes or shapes != set(reference["shapes"]):
        return False
    inputs = []
    for artifact in record.artifacts:
        if artifact.role == "evaluation_input":
            value = json.loads(records.read_artifact(artifact))
            if isinstance(value, dict) and value.get("schema_version") == 1 and value.get("kind") == "run":
                inputs.append(value)
    covered = set()
    for step in record.steps:
        identity = step.measurement_identity
        if identity is None or identity.method not in {"atrex-bench/eval", "atrex-bench/abba"} or step.hardware != hardware:
            return False
        source = {key: value for key, value in identity.parameters.items()
                  if key.startswith(("candidate_source", "baseline_source"))}
        matched = False
        for value in inputs:
            request = value.get("request")
            if not isinstance(request, dict):
                continue
            spec = request.get("spec", {})
            target_matches = (spec.get("target_hardware") == [hardware] if "target_hardware" in spec
                              else spec.get("gpu") == hardware)
            if not target_matches:
                continue
            prepared = gateway_identity(request, value.get("reference_files", {}), source_parameters=source)
            if prepared.input_digest != identity.input_digest:
                continue
            batch = request.get("reference", {}).get("shapes", {})
            if not isinstance(batch, dict) or not batch or not set(batch) <= shapes:
                return False
            expected = shape_batch_request({"reference": reference}, list(batch))["reference"]
            if any(request["reference"].get(field) != data for field, data in expected.items()):
                return False
            if {point.shape_id for point in identity.points} != set(batch):
                return False
            covered.update(batch)
            matched = True
            break
        if not matched:
            return False
    return covered == shapes


def measurement_evidence(records, record, *, owner, task, hardware):
    """Reuse native correctness / ABBA projection, adding only current Task scope."""
    view = RecordedMeasurements(workspace_provider=owner, evaluation_input=task)
    evidence = view.project(records, record)
    result = evidence["result"]
    if (evidence["execution_status"] != "completed" or result.get("correct") is not True
            or result.get("error") or result.get("failure") or result.get("qualification")
            or not _full_task(records, record, task, hardware)):
        return None
    measurements = view._measurements(record)
    if len({item.identity.method for item in measurements}) != 1:
        return None
    if all(item.identity.method == "atrex-bench/eval" for item in measurements):
        repetitions = {item.identity.parameters.get("measurement_repetitions") for item in measurements}
        points = [(point.side, point.repeat, point.shape_id) for item in measurements for point in item.identity.points]
        shapes = {point[2] for point in points}
        if (len(repetitions) != 1 or type(count := next(iter(repetitions))) is not int or count < 1
                or len(points) != len(set(points))
                or set(points) != {("candidate", repeat, shape) for repeat in range(count) for shape in shapes}):
            return None
    modes = {item.identity.parameters.get("mode") for item in measurements}
    if modes not in ({"full"}, {"correctness_only"}):
        return None
    # RecordedMeasurements owns the complete native schedule and run verdicts.
    # Correctness-only evidence never acquires performance from a stray scalar.
    performance = (modes == {"full"} and positive(result.get("latency_us_geomean"))
        and all(positive(value) for value in result.get("latency_us_by_shape", {}).values())
        and set(result.get("latency_us_by_shape", {})) == {
            point.shape_id for item in measurements for point in item.identity.points})
    comparison = result.get("comparison") if result.get("complete") is True else None
    baselines = {item.identity.parameters.get("baseline_source_sha256") for item in measurements}
    return {"source_identity": evidence["source_sha256"], "performance": performance,
            "measurements": measurements, "latency_us": result.get("latency_us_geomean"),
            "comparison": comparison, "baseline_sha256": next(iter(baselines)) if len(baselines) == 1 else None,
            "baseline_latency_us": result.get("baseline", {}).get("latency_us_geomean"),
            "candidate_sha256": measurements[0].identity.parameters.get("candidate_source_artifact_sha256")}


def comparison_key(measurements):
    """Keep the existing common contract for comparisons between records."""
    from .run_state import digest
    if (not measurements or any(item.comparability != "comparable" or not item.environment_identity
                               for item in measurements)
            or len({item.environment_identity for item in measurements}) != 1):
        return None
    signatures = []
    for item in measurements:
        value = item.identity.as_dict()
        value.pop("input_digest")
        value["parameters"] = {key: val for key, val in value["parameters"].items()
            if not key.startswith("candidate_source") and key != "correctness_digest"}
        value["points"] = [{key: val for key, val in point.items() if key != "revision"} for point in value["points"]]
        signatures.append({"identity": value, "environment": item.environment_identity})
    return digest(signatures)


def needs_new_hypothesis(hypothesis):
    # Unknown is not measured stagnation, nor an unlimited keep allowance.
    return hypothesis.get("stagnant", 0) >= 2 or hypothesis.get("uncertain_windows", 0) >= 2
