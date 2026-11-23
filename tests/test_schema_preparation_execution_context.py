"""Public fresh preparation selects its own CPU receipt, without discovery."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from aka.contracts.execution import ExecutionFacts, JobResult
from aka.contracts.execution_environment import EnvironmentSnapshot
from aka.contracts.execution_recovery import JobReference
from aka.execution.legacy.executor import LegacyExecutor
from aka.task.candidate_workspace.provider import GitCandidateWorkspace
from aka.workflow.schema._native import execution_context
from aka.workflow.schema.execution_context import CORRECTABLE, sha256
from aka.workflow.schema.materials import DesignRef, read_design
from aka.workflow.schema.run_state import digest
from test_schema_native_execution_context import retained_native


@pytest.fixture
def prepared_native(retained_native, monkeypatch):
    h = retained_native
    h.campaign.candidate_workspace = GitCandidateWorkspace()
    source = h.records.read_source(next(a for a in h.record.artifacts if a.role == "candidate-source"))
    h.prepared["source_identity"] = h.campaign.candidate_workspace.source_identity(source)
    monkeypatch.setattr(h.executor, "query_environment", lambda q: pytest.fail("unexpected native discovery"))
    return h


def automatic(h):
    return execution_context(campaign=h.campaign, prepared=h.prepared, executor=h.executor,
        directory=h.directory, preparation_records=(h.record.ref,))


def test_preparation_original_job_is_stronger_than_cache_without_task_review(prepared_native):
    h = prepared_native
    before = digest(h.records.get(h.record.ref).as_dict())
    document = automatic(h).as_dict()
    assert document["architecture"] == {"value": "sm_103", "source": "observed",
        "evidence_sha256": h.evidence.raw_sha256}
    assert document["toolchain"]["value"] == "3.7.1"
    assert document["toolchain"]["source"] == "observed"
    assert document["binding"]["native_policy_sha256"] == digest(h.prepared["native_policy"])
    assert all(document["task_constraints"][path] == {"status": "unknown", "value": None}
               for path in CORRECTABLE)
    assert digest(h.records.get(h.record.ref).as_dict()) == before
    assert h.record.attempts[0].result.environment_snapshot.source == "cache"


def revised_record(h, monkeypatch, record):
    original = h.records.get
    monkeypatch.setattr(h.records, "get", lambda ref: record if ref == h.record.ref else original(ref))


@pytest.mark.parametrize("damage", ["run", "task", "task-bytes", "prepared-source", "receipt-source", "route", "namespace",
                                   "framework", "job-owner", "architecture"])
def test_conflicting_preparation_receipt_cannot_gain_authority(prepared_native, monkeypatch, damage):
    h = prepared_native
    if damage == "run":
        h.campaign.workspace = h.campaign.workspace / "another-run"
    elif damage == "task":
        h.campaign.problem.evaluation = replace(h.campaign.problem.evaluation, case_ids=("other",))
    elif damage == "task-bytes":
        task = h.campaign.problem.evaluation
        h.campaign.problem.evaluation = replace(task, files=tuple(
            replace(item, text=item.text + "\n# another complete Task\n")
            if item.path == "reference.py" else item for item in task.files))
    elif damage == "prepared-source":
        h.prepared["source_identity"] = "0" * 64
    elif damage == "route":
        h.executor.configuration["url"] = "https://other.invalid"
        h.executor.request = replace(h.executor.request, url="https://other.invalid")
    elif damage == "namespace":
        h.executor.request = replace(h.executor.request, environment={"AGATE_AK": "other-owner"})
    elif damage == "framework":
        h.prepared["native_policy"]["framework"] = "Cuda"
    else:
        attempt = h.record.attempts[-1]
        body = json.loads(attempt.result.stdout)
        if damage == "receipt-source":
            body["result"]["shape_selection"]["source_hashes"]["candidate.py"] = "0" * 64
        elif damage == "job-owner":
            body["job_id"] = "another-owned-job"
        else:
            body["result"]["environment"]["gpu_arch"] = "sm_89"
        revised_record(h, monkeypatch, replace(h.record, attempts=(replace(attempt,
            result=replace(attempt.result, stdout=json.dumps(body))),)))
    with pytest.raises(ValueError):
        automatic(h)


@pytest.mark.parametrize("explicit", [False, True], ids=["automatic", "explicit"])
def test_known_native_toolchain_conflict_is_rejected(prepared_native, monkeypatch, explicit):
    h = prepared_native
    attempt = h.record.attempts[-1]
    body = json.loads(attempt.result.stdout)
    body["result"]["environment"]["triton_version"] = "3.6.0"
    raw = json.dumps(body).encode()
    record = replace(h.record, attempts=(replace(attempt, result=replace(attempt.result, stdout=raw.decode())),))
    revised_record(h, monkeypatch, record)
    with pytest.raises(ValueError):
        if explicit:
            evidence = replace(h.evidence, raw=raw, raw_sha256=sha256(raw), record_sha256=digest(record.as_dict()))
            execution_context(campaign=h.campaign, prepared=h.prepared, executor=h.executor,
                directory=h.directory, retained_job=evidence)
        else:
            automatic(h)


def test_latest_preparation_stdout_cannot_use_an_older_attempts_job(prepared_native, monkeypatch):
    h = prepared_native
    attempt = h.record.attempts[-1]
    original, = attempt.result.job_references
    latest = replace(attempt, attempt_id="synthetic-new-attempt",
        result=replace(attempt.result, job_references=(replace(original, identifier="synthetic-new-job"),)))
    revised_record(h, monkeypatch, replace(h.record, attempts=(*h.record.attempts, latest)))
    with pytest.raises(ValueError):
        automatic(h)


@pytest.mark.parametrize("absence", ["no-receipt", "non-json", "no-job-owner", "no-association",
                                    "failed", "partial", "partial-assessment", "failed-assessment"])
def test_unqualified_preparation_keeps_legal_cache_fallback(prepared_native, monkeypatch, absence):
    h = prepared_native
    attempt = h.record.attempts[-1]
    result = attempt.result
    body = json.loads(result.stdout)
    if absence == "no-receipt":
        result = replace(result, stdout="")
    elif absence == "non-json":
        result = replace(result, stdout="legacy textual receipt")
    elif absence == "no-job-owner":
        result = replace(result, job_references=())
    elif absence == "failed":
        body["status"] = "failed"
        result = replace(result, stdout=json.dumps(body), transport_status="failed", exit_code=1)
    elif absence == "partial":
        body["result"]["shape_selection"]["partial"] = True
        result = replace(result, stdout=json.dumps(body))
    elif absence == "failed-assessment":
        attempt = replace(attempt, assessment=replace(attempt.assessment, validity="invalid"))
    elif absence == "partial-assessment":
        assessment = attempt.assessment
        attempt = replace(attempt, assessment=replace(assessment, completeness="partial",
            observed=assessment.observed[:-1], missing=assessment.observed[-1:]))
    record = replace(h.record, attempts=(replace(attempt, result=result),))
    if absence == "no-association":
        record = replace(record, artifacts=tuple(a for a in record.artifacts if a.role != "live_eval_association"))
    revised_record(h, monkeypatch, record)
    document = automatic(h).as_dict()
    assert document["architecture"]["source"] == document["toolchain"]["source"] == "cache"
    assert document["architecture"]["evidence_sha256"] != h.evidence.raw_sha256


def test_snapshot_fallback_retains_native_record_precedence(prepared_native, monkeypatch):
    h = prepared_native
    attempt = h.record.attempts[-1]
    revised_record(h, monkeypatch, replace(h.record, attempts=(replace(attempt,
        result=replace(attempt.result, stdout="")),)))
    older = replace(h.record.ref, experiment_id="0" * 64)
    document = execution_context(campaign=h.campaign, prepared=h.prepared, executor=h.executor,
        directory=h.directory, preparation_records=(older, h.record.ref)).as_dict()
    assert document["architecture"]["source"] == "cache"


def install_recorded_preparation(harness, monkeypatch):
    """Fake only the CPU evaluator/transport below actual public preparation."""
    from aka.evaluation.gateway_command import GatewayCommandEvaluator
    from aka.execution.gateway.requests import operation_request
    from aka.execution.live_eval import WIRE_PATH, digest as wire_digest
    from aka.experiment.file_records import FileExperimentRecords
    from aka.experiment.runner import LegacyExperiment
    from schema_cpu_fixture import CpuCampaign
    from test_native_environment_compat import request, raw_evaluation

    original_init = CpuCampaign.__init__
    def initialize(self, root):
        original_init(self, root)
        _, self.problem.evaluation = request(root / "native")
    monkeypatch.setattr(CpuCampaign, "__init__", initialize)
    route = SimpleNamespace(backend="legacy", endpoint="https://cpu.invalid", ssh=False,
                            environment={"AGATE_AK": "synthetic-namespace"})
    monkeypatch.setattr(LegacyExecutor, "_environment_backend", lambda self: route)
    monkeypatch.setattr(LegacyExecutor, "query_environment", lambda *a: pytest.fail("public startup queried environment"))
    receipts = []
    def measure(campaign, workspace, argv, **kwargs):
        command, _ = request(workspace)
        body = json.loads(command.submission)
        body["request"]["candidate"] = (workspace / "kernel.py").read_text()
        command = replace(command, submission=json.dumps(body).encode(),
            environment=route.environment, idempotency_key="synthetic-public-v0")
        records = FileExperimentRecords(workspace / ".cpu-v0-records", str(campaign.workspace.absolute()))
        runtime = campaign._supervisor_runtime
        runtime.read_experiment_records = lambda: tuple(records.iter_records())
        runtime.read_experiment_source = records.read_source
        runtime._record_reader = lambda **kwargs: nullcontext(records)
        def execute(command):
            record, = records.iter_records()
            attempt = record.attempts[-1]
            original = json.loads(records.read_artifact(next(a for a in record.artifacts if a.role == "evaluation_input")))
            wire = operation_request(original["request"], "run", ssh=False)
            wire["idempotency_key"] = command.idempotency_key
            wire.setdefault("options", {})["timeout_s"] = command.timeout
            ref = JobReference("gateway", "synthetic-public-v0", endpoint=route.endpoint,
                request_identity=wire_digest(wire), idempotency_key=command.idempotency_key)
            records.publish_content(record.ref, json.dumps({"attempt_id": attempt.attempt_id,
                "terminal": True, "job_reference": {"identifier": ref.identifier},
                "request_identity": ref.request_identity,
                "route_identity": wire_digest([route.endpoint, WIRE_PATH, "", ""]),
                "namespace_identity": wire_digest(route.environment["AGATE_AK"])}).encode(), "live_eval_association")
            hashes = {name: sha256(value.encode()) for name, value in original["reference_files"].items()
                if name in {"input.py", "reference.py", "metadata.json", "shapes.json"}}
            hashes["candidate.py"] = sha256(wire["candidate"].encode())
            raw = json.dumps({"job_id": ref.identifier, "status": "succeeded", "num_gpus": 1,
                "request_digest": {"gpu": "NVIDIA L20D", "num_gpus": 1},
                "result": {**raw_evaluation(), "dsl": "triton",
                    "environment": {"gpu_name": "NVIDIA L20D", "gpu_arch": "sm_103", "triton_version": "3.7.1"},
                    "shape_selection": {"all_shape_ids": list(campaign.problem.evaluation.case_ids),
                        "partial": False, "source_hashes": hashes}}})
            receipts.append(raw.encode())
            snapshot = EnvironmentSnapshot("legacy", route.endpoint, "L20D", None,
                observed={"gpu": "L20D", "arch": "sm_103", "toolchain": {"triton": "3.7.1"}}, source="cache")
            return JobResult(0, stdout=raw, transport_status="succeeded",
                facts=ExecutionFacts(disposition="terminal"), environment_snapshot=snapshot, job_references=(ref,))
        experiment = LegacyExperiment(SimpleNamespace(capabilities=LegacyExecutor.capabilities, run=execute),
            records=records, command_evaluator=GatewayCommandEvaluator(
                execution_capabilities=LegacyExecutor.capabilities, evaluation_input=campaign.problem.evaluation))
        record = experiment.run_recorded(command, request_id="synthetic-public-v0")
        result = experiment.command_result(record.ref)
        return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    monkeypatch.setattr(CpuCampaign, "measure_for_acceptance", measure)
    return receipts


@pytest.mark.parametrize("paired", [False, True], ids=["fresh-C", "import-first-E"])
def test_public_cli_uses_own_complete_v0_before_first_planner(tmp_path, monkeypatch, paired):
    from aka.bootstrap import workflow_cli
    from schema_public_fixture import install_native_cpu, schema_config
    harness = install_native_cpu(monkeypatch, tmp_path)
    receipts = install_recorded_preparation(harness, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_MODEL", "qwen3.8-max[1m]")
    configuration = tmp_path / "schema.json"
    configuration.write_text(json.dumps(schema_config(harness, paired=paired, episodes=1)))
    patch = tmp_path / "native.patch.json"
    patches = [{"id": "resources", "config": {"plugins": [], "bundles": []}},
        {"id": "workflow", "config": {"startup": True, "search": {}, "inherit_state": False,
            "output_limit_recovery_retries": 0, "episode_timeout_seconds": 60}}]
    if paired:
        patches.append({"id": "source", "config": {"kind": "kernel", "kernel_demo": str(tmp_path / "input.py")}})
    patch.write_text(json.dumps({"api_version": 1, "patches": patches}))
    (tmp_path / "plugins/gpu-wiki").mkdir(parents=True)
    (tmp_path / "gpu-wiki").mkdir()
    arguments = ["--repo-root", str(tmp_path), "--profile", "light", "--patch", str(patch),
        "--problem", str(harness.task), "--workspace-base", str(tmp_path / "runs"),
        "--workspace-suffix", "fresh_receipt", "--platform", "cpu", "--framework", "triton",
        "--max-iters", "0", "--target-util", "0", "--fast-episodes", "0", "--handoff-resumes", "0",
        "--hardware", "L20D", "--schema", "--schema-config", str(configuration)]
    assert workflow_cli.main(arguments) == 0
    campaign, = harness.campaigns
    state = campaign.driver.state.data
    assert [row["kind"] for row in harness.calls] == ["optimize" if paired else "design", "implement"]
    assert len(receipts) == campaign.setup_count == 1
    assert state["counts"]["agent_calls"] == 2 and state["counts"]["episodes"] == 1
    frozen = state["planner_request"]
    context = json.loads(frozen["execution_context"])
    assert context["architecture"] == {"value": "sm_103", "source": "observed", "evidence_sha256": sha256(receipts[0])}
    assert context["toolchain"]["source"] == "observed" and context["toolchain"]["value"] == "3.7.1"
    assert all(item["status"] == "unknown" for item in context["task_constraints"].values())
    assert "contract_correction_sha256" not in frozen
    artifact = read_design(DesignRef.from_dict(state["design"]))
    assert b"(cache)" not in artifact.checked.files["root.yaml"]
    if paired:
        parent = read_design(DesignRef.from_dict(artifact.manifest["parent"]))
        assert parent.manifest["mode"] == "import"
    # A durable context is restored without re-reading mutable discovery or
    # native receipts; original operation hashes and native accounting persist.
    before = deepcopy(state)
    campaign.experiment_record_reader = lambda: pytest.fail("frozen context reread preparation")
    campaign.driver._recovery_executor.query_environment = lambda *a: pytest.fail("frozen context queried")
    assert campaign.driver._new_context(campaign, "another-operation").sha256 == frozen["execution_context_sha256"]
    assert state == before
