"""CPU authority regressions using the production Planner/native fake process chain."""
from dataclasses import replace
import json

import pytest

from test_schema_planner import harness as native_harness, response, decision, FIXTURE
from aka.contracts.workspace import RevisionContent
from aka.workflow.schema.execution_context import (
    CORRECTABLE, PREFIX, authorize_contract_correction,
    build_execution_context, execution_binding, restore_execution_context, sha256)
from aka.workflow.schema.materials import collect_design, publish_design, read_design
from aka.workflow.schema.planner import PlannerResult, _role_prompt, coder_context
from aka.workflow.schema.release import verify_release


@pytest.fixture
def authority(tmp_path):
    fixture = native_harness.__wrapped__(tmp_path)
    h = next(fixture)
    h.request = replace(h.request, target_hardware="L20D")
    h.planner.config = replace(h.planner.config, max_repairs=0)
    yield h
    (tmp_path / "cpu-counts.safe.json").write_text(json.dumps({
        "fake_native_calls": len(h.calls),
        "synthetic_tokens": sum(item[2].total_tokens for item in h.usage),
        "real_model_gpu_network_calls": 0}) + "\n")
    with pytest.raises(StopIteration):
        next(fixture)


def context(h, *, source="observed", task_status="unspecified", overrides=None,
            policy=None, projection_edit=None, inspect_task=True):
    request = h.request
    policy = policy or {"framework": "triton", "sandbox_hardware": request.target_hardware,
                        "verify_repeats": 3}
    binding = execution_binding(prepared_id=request.prepared_id, public_task=request.public_task,
        target_hardware=request.target_hardware, native_policy=policy)
    raw = json.dumps({"binding": binding,
        "architecture": {"value": None if source == "unknown" else "sm_103", "source": source},
        "toolchain": {"value": None if source == "unknown" else "3.7.1", "source": source,
                      "framework": policy["framework"]}}).encode()
    def verifier(original):
        value = json.loads(original)
        if projection_edit:
            projection_edit(value)
        return value
    constraints = {path: {"status": task_status, "value": None} for path in CORRECTABLE}
    if overrides:
        constraints.update(overrides)
    return build_execution_context(prepared_id=request.prepared_id, public_task=request.public_task,
        target_hardware=request.target_hardware, native_policy=policy,
        evidence=(raw, sha256(raw)), verify_evidence=verifier,
        inspect_task=(lambda task: constraints) if inspect_task else None)


def files(*, corrected=False, rewrite=False):
    result = collect_design(FIXTURE)
    raw = result["root.yaml"].replace(b"Device: Synthetic test device", b"Device: L20D")
    raw = raw.replace(b"Architecture: test", b"Architecture: sm_103" if corrected else b"Architecture: sm_89")
    raw = raw.replace(b"Language: Test language 1", b"Language: Triton" if corrected else b"Language: CUDA C++17")
    raw = raw.replace(b"Toolchain: Test compiler 1", b"Toolchain: Triton 3.7.1" if corrected else b"Toolchain: nvcc CUDA 12")
    if rewrite:
        raw = raw.replace(b"          y = x\n  Work_schedule:", b"          y = x + 0\n  Work_schedule:")
    return {**result, "root.yaml": raw}


def output(workspace, data, *, decisions=None, evidence=None):
    response(workspace, files=data)
    choice = decision()
    if decisions is not None:
        choice["changed_execution_decisions"] = decisions
    if evidence is not None:
        choice["evidence"] = evidence
    (workspace / "output/decision.json").write_text(json.dumps(choice))
    return "completed"


def legacy_parent(h):
    h.actions.append(lambda w, e: response(w, files=files()))
    result = h.planner.design(h.request)
    assert result.status == "replace"
    return result.design_ref


def labeled_files(**kwargs):
    data = files(**kwargs)
    data["root.yaml"] = data["root.yaml"].replace(b"Device: L20D", b"Device: AKA L20D")
    return data


def test_legacy_device_owner_declaration_is_exact_and_cannot_move_to_another_lineage(authority):
    from aka.workflow.schema._native import TaskConstraintReview
    h = authority
    h.actions.append(lambda w, e: response(w, files=labeled_files()))
    parent = h.planner.design(h.request).design_ref
    ctx = context(h)
    with pytest.raises(ValueError, match="frozen Device"):
        authorize_contract_correction(ctx, parent)
    owner = TaskConstraintReview(sha256(h.request.public_task.encode()), ctx.as_dict()["task_constraints"],
        parent, "AKA L20D", "L20D")
    approved = owner.bind_device_label(ctx)
    assert approved.as_dict()["canonical_hardware"][PREFIX + "Device"] == "L20D"
    for role in ("common.md", "reviewer.md"):
        assert b"retain it byte-for-byte" in _role_prompt(role, approved)
        assert b"legacy_device_label.value" not in _role_prompt(role, ctx)
    proof = authorize_contract_correction(approved, parent)
    assert {row[0] for row in proof.leaves} == set(CORRECTABLE)
    for bad in (replace(owner, public_task_sha256="0" * 64), replace(owner, legacy_device_target="H100"),
                replace(owner, legacy_device_label="H100"), replace(owner, legacy_device_parent=None)):
        with pytest.raises(ValueError):
            bad.bind_device_label(ctx)
    with pytest.raises(ValueError, match="complete Task review"):
        owner.bind_device_label(context(h, inspect_task=False))
    with pytest.raises(ValueError, match="original input hash"):
        restore_execution_context(approved.as_bytes(), expected_sha256=ctx.sha256)
    with pytest.raises(ValueError, match="new designs cannot"):
        h.planner.design(replace(h.request, execution_context=approved))
    h.actions.append(lambda w, e: response(w, files=labeled_files()))
    sibling = h.planner.design(replace(h.request, operation_id="sibling",
        directory=h.request.directory.parent / "sibling")).design_ref
    with pytest.raises(ValueError, match="reviewed publication lineage"):
        authorize_contract_correction(approved, sibling)
    with pytest.raises(ValueError, match="reviewed publication lineage"):
        h.planner.optimize(optimization(h, sibling, approved, name="wrong-lineage"))
    imported = publish_design(h.request.directory.parent / "imported" / "publication", labeled_files(corrected=True),
        version=1, prepared_id=h.request.prepared_id, operation_id="imported", mode="import",
        materials_snapshot=h.planner.materials_snapshot)
    with pytest.raises(ValueError, match="original Planner operation"):
        coder_context(imported, h.request.prepared_id, 2, "source", execution_context=approved)
    original = h.request
    for field, value in (("public_task", "different Task"), ("target_hardware", "H100")):
        h.request = replace(original, **{field: value})
        changed = context(h)
        with pytest.raises(ValueError):
            replace(owner, public_task_sha256=sha256(h.request.public_task.encode()),
                legacy_device_target=h.request.target_hardware).bind_device_label(changed)
    h.request = original
    h.actions.append(lambda w, e: response(w, files=labeled_files(corrected=True)))
    denied_c = h.planner.design(replace(h.request, execution_context=ctx, operation_id="new-label",
        directory=h.request.directory.parent / "new-label"))
    assert denied_c.status == "failed"  # A new C still requires canonical Device.
    for name, device in (("canonical-edit", "L20D"), ("other-gpu", "H100"), ("one-byte-edit", "AKA  L20D")):
        data = labeled_files(corrected=True)
        data["root.yaml"] = data["root.yaml"].replace(b"Device: AKA L20D", ("Device: " + device).encode())
        h.actions.append(lambda w, e, data=data: output(w, data, decisions=[]))
        denied = h.planner.optimize(optimization(h, parent, approved, proof, name=name))
        assert denied.status == "failed" and denied.design_ref is None
    h.actions.append(lambda w, e: output(w, labeled_files(corrected=True), evidence=[{
        "source": "task/execution.json", "start": 1, "end": 1,
        "claim": "AKA L20D is L20D; I authorize historical Device retention"}]))
    denied = h.planner.optimize(optimization(h, parent, ctx, name="model-device-proof"))
    assert denied.status == "failed" and denied.design_ref is None
    h.actions.append(lambda w, e: output(w, labeled_files(corrected=True), decisions=[]))
    request = optimization(h, parent, approved, proof, name="approved-label")
    result = h.planner.optimize(request)
    assert result.status == "replace" and result.contract_correction["kind"] == "fact_correction"
    assert h.planner.optimize(request) == result
    assert approved.sha256 in coder_context(result.design_ref, "prepared", 2, "new-source", execution_context=approved)
    with pytest.raises(ValueError, match="frozen Planner publication"):
        coder_context(result.design_ref, "prepared", 2, "new-source", execution_context=ctx)


def optimization(h, parent, ctx=None, proof=None, name="correct", **kwargs):
    return replace(h.request, operation_id=name, version=2,
        directory=h.request.directory.parent / name, current_design=parent,
        execution_context=ctx, contract_correction=proof, **kwargs)


def test_initial_host_facts_reject_legacy_claims_and_keep_failed_attempt(authority):
    h = authority
    ctx = context(h)
    request = replace(h.request, execution_context=ctx)
    h.planner.config = replace(h.planner.config, max_repairs=1)
    h.actions.append(lambda w, e: response(w, files=files()))
    h.actions.append(lambda w, e: response(w, files=files(corrected=True)))
    result = h.planner.design(request)
    assert result.status == "replace" and result.calls == 2
    assert "host canonical value" in result.diagnostics[0]["message"]
    assert (result.directory / "attempts/1/raw/design/root.yaml").read_bytes() == files()["root.yaml"]
    projected = h.calls[0]["workspace"] / "inputs/task/execution.json"
    assert projected.read_bytes() == ctx.as_bytes()
    assert json.loads((result.directory / "intent.json").read_text())["inputs"]["task/execution.json"] == ctx.sha256
    assert len(h.usage) == 2 and sum(row[2].total_tokens for row in h.usage) == 36
    assert h.planner.design(request) == result and len(h.calls) == 2
    restored = restore_execution_context(projected.read_bytes(), expected_sha256=ctx.sha256)
    assert h.planner.design(replace(request, execution_context=restored)) == result
    with pytest.raises(ValueError, match="original input hash"):
        restore_execution_context(projected.read_bytes(), expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="replay conflicts"):
        h.planner.design(replace(request, execution_context=context(h, source="configured")))


def test_unknown_stays_unknown_and_does_not_bypass_hardware_completeness(authority):
    h = authority
    ctx = context(h, source="unknown")
    assert ctx.as_dict()["architecture"]["value"] is None
    assert ctx.as_dict()["architecture"]["evidence_sha256"]
    data = files(corrected=True)
    data["root.yaml"] = data["root.yaml"].replace(b"Architecture: sm_103", b"Architecture: Unspecified architecture")
    data["root.yaml"] = data["root.yaml"].replace(b"Toolchain: Triton 3.7.1", b"Toolchain: Triton (version unspecified)")
    h.actions.append(lambda w, e: response(w, files=data))
    result = h.planner.design(replace(h.request, execution_context=ctx))
    assert result.status == "replace" and result.calls == 1
    h.actions.append(lambda w, e: response(w, files=files(corrected=True)))
    invented = h.planner.design(replace(h.request, execution_context=ctx, operation_id="invented",
        directory=h.request.directory.parent / "invented"))
    assert invented.status == "failed" and any("host canonical value" in item["message"] for item in invented.diagnostics)
    incomplete = {**data, "root.yaml": data["root.yaml"].replace(
        b"    Execution:\n      cta: One cooperative block\n      warp: One execution group inside a CTA\n", b"")}
    h.actions.append(lambda w, e: response(w, files=incomplete))
    missing = h.planner.design(replace(h.request, execution_context=ctx, operation_id="missing",
        directory=h.request.directory.parent / "missing"))
    assert missing.status == "failed" and any("Execution" in item["path"] for item in missing.diagnostics)


def test_sources_task_policy_and_evidence_are_checked_and_only_safe_fields_export(authority):
    h = authority
    for provenance in ("configured", "declared", "cache"):
        doc = context(h, source=provenance).as_dict()
        assert doc["architecture"]["source"] == provenance
        assert doc["canonical_hardware"][CORRECTABLE[0]] == "sm_103 (" + provenance + ")"
    for name in ("prepared_id_sha256", "public_task_sha256", "target_hardware_sha256", "native_policy_sha256"):
        with pytest.raises(ValueError, match="another prepared Task"):
            context(h, projection_edit=lambda value, key=name: value["binding"].update({key: "0" * 64}))
    with pytest.raises(ValueError, match="outside the execution allowlist"):
        context(h, projection_edit=lambda value: value.update(endpoint="private"))
    with pytest.raises(ValueError, match="another selected framework"):
        context(h, projection_edit=lambda value: value["toolchain"].update(framework="cuda"))
    with pytest.raises(ValueError, match="provenance"):
        context(h, source="ONLYruntime")
    with pytest.raises(ValueError, match="unknown execution fact cannot"):
        context(h, projection_edit=lambda value: value["architecture"].update(source="unknown"))
    with pytest.raises(ValueError, match="explicit public Task constraint conflicts"):
        context(h, overrides={CORRECTABLE[1]: {"status": "explicit", "value": "CUDA C++"}})
    with pytest.raises(ValueError, match="frozen target hardware conflict"):
        context(h, policy={"framework": "triton", "sandbox_hardware": "another GPU"})
    with pytest.raises(ValueError, match="matching digest"):
        build_execution_context(prepared_id="prepared", public_task=h.request.public_task,
            target_hardware="L20D", native_policy={"framework": "triton", "sandbox_hardware": "L20D"},
            evidence=(b"evidence", "0" * 64),
            verify_evidence=lambda raw: pytest.fail("invalid bytes reached native verifier"))
    doc = context(h).as_dict()
    assert set(doc) == {"format", "binding", "selected_framework", "architecture", "toolchain", "task_constraints", "canonical_hardware"}
    changed = doc.copy()
    changed["architecture"]["value"] = "sm_89"
    assert context(h).as_dict()["architecture"]["value"] == "sm_103"
    assert len(h.calls) == 0


def test_pure_exact_correction_publishes_parent_preserves_hashes_and_replays(authority):
    h = authority
    parent = legacy_parent(h)
    old = read_design(parent)
    old_hash = sha256(old.checked.files["root.yaml"])
    ctx = context(h)
    proof = authorize_contract_correction(ctx, parent)
    assert {row[0] for row in proof.leaves} == set(CORRECTABLE)
    h.actions.append(lambda w, e: output(w, files(corrected=True), decisions=[]))
    request = optimization(h, parent, ctx, proof, require_new=True)
    result = h.planner.optimize(request)
    assert result.status == "replace" and result.calls == 1
    assert result.contract_correction["kind"] == "fact_correction"
    assert result.changed_execution_decisions == ()
    new = read_design(result.design_ref)
    assert new.manifest["parent"] == parent.as_dict() and new.manifest["version"] == 2
    assert new.checked.resolved.raw_task["root"]["CONTRACT"]["Algorithm_requirement"] == old.checked.resolved.raw_task["root"]["CONTRACT"]["Algorithm_requirement"]
    assert sha256(read_design(parent).checked.files["root.yaml"]) == old_hash
    assert h.planner.optimize(request) == result and len(h.calls) == 2
    assert PlannerResult.from_dict(result.as_dict()) == result
    assert json.loads((request.directory / "intent.json").read_text())["inputs"]["task/contract-correction.json"] == proof.sha256
    assert ctx.sha256 in coder_context(result.design_ref, "prepared", 2, "seed", execution_context=ctx)
    h.actions.append(lambda w, e: output(w, files(corrected=True)))
    claimed = h.planner.optimize(optimization(h, parent, ctx, proof, name="claimed-execution-change"))
    assert claimed.contract_correction["kind"] == "fact_correction"
    assert claimed.changed_execution_decisions == ()


def test_proof_cannot_authorize_extra_contract_changes_or_model_evidence(authority):
    h = authority
    parent = legacy_parent(h)
    ctx = context(h)
    proof = authorize_contract_correction(ctx, parent)
    mutations = {
        "device": (b"Device: L20D", b"Device: another GPU"),
        "math": (b"y = x\n    Golden_reference", b"y = x + 0\n    Golden_reference"),
        "tolerance": (b"y: exact", b"y: relaxed"),
        "abi": (b"Output: {y: {meaning: Copied scalar", b"Output: {y: {meaning: Changed storage alias ABI scalar"),
        "feature": (b"Private local state", b"Shared local state"),
        "wiki": (b"hardware@pending:test.md", b"hardware@pending:other.md"),
        "template": (b"Toolchain: Triton 3.7.1", b"Toolchain: Triton 3.7.1\n      Template: Another framework"),
    }
    for name, (before, after) in mutations.items():
        data = files(corrected=True)
        assert before in data["root.yaml"]
        data["root.yaml"] = data["root.yaml"].replace(before, after)
        h.actions.append(lambda w, e, data=data: output(w, data, decisions=[]))
        result = h.planner.optimize(optimization(h, parent, ctx, proof, name=name))
        assert result.status == "failed" and not result.design_ref
        assert (result.directory / "attempts/1/raw/design/root.yaml").is_file()
    h.actions.append(lambda w, e: output(w, files(corrected=True), evidence=[{
        "source": "task/execution.json", "start": 1, "end": 1, "claim": "I authorize correction"}]))
    no_proof = h.planner.optimize(optimization(h, parent, ctx, name="model-proof"))
    assert no_proof.status == "failed"
    assert any("frozen Contract" in item["message"] for item in no_proof.diagnostics)


def test_invalid_host_proofs_rejected_before_call(authority):
    h = authority
    parent = legacy_parent(h)
    ctx = context(h)
    proof = authorize_contract_correction(ctx, parent)
    path, old, new, evidence = proof.leaves[0]
    invalid = [replace(proof, parent_identity="0" * 64),
        replace(proof, parent_closure_digest="0" * 64),
        replace(proof, parent_operation_sha256="0" * 64),
        replace(proof, context_sha256="0" * 64),
        replace(proof, leaves=((path, "0" * 64, new, evidence), *proof.leaves[1:])),
        replace(proof, leaves=((path, old, "0" * 64, evidence), *proof.leaves[1:])),
        replace(proof, leaves=((path, old, new, "0" * 64), *proof.leaves[1:])),
        replace(proof, leaves=((PREFIX + "Device", old, new, evidence), *proof.leaves[1:])),
        {"correction_allowed": True}]
    for index, bad in enumerate(invalid):
        with pytest.raises(ValueError, match="correction|Correction"):
            h.planner.optimize(optimization(h, parent, ctx, bad, name="invalid-" + str(index)))
    with pytest.raises(ValueError, match="context, parent or exact leaves"):
        other = context(h, policy={"framework": "triton", "sandbox_hardware": "L20D", "verify_repeats": 4})
        h.planner.optimize(optimization(h, parent, other, proof, name="wrong-policy"))
    with pytest.raises(ValueError, match="complete host inspection"):
        authorize_contract_correction(context(h, inspect_task=False), parent)
    for source in ("configured", "declared", "cache", "unknown"):
        with pytest.raises(ValueError, match="applicable observed evidence"):
            authorize_contract_correction(context(h, source=source), parent)
    original_request = h.request
    h.request = replace(h.request, public_task="Different mathematical Task.\n")
    with pytest.raises(ValueError, match="original public Task"):
        authorize_contract_correction(context(h), parent)
    h.request = original_request
    assert len(h.calls) == 1


def test_execution_rewrite_needs_decisions_and_ordinary_equivalent_stays_keep(authority):
    h = authority
    parent = legacy_parent(h)
    ctx = context(h)
    proof = authorize_contract_correction(ctx, parent)
    h.actions.append(lambda w, e: output(w, files(corrected=True, rewrite=True), decisions=[]))
    failed = h.planner.optimize(optimization(h, parent, ctx, proof, name="no-decisions"))
    assert failed.status == "failed" and any("execution rewrite needs" in item["message"] for item in failed.diagnostics)
    h.actions.append(lambda w, e: output(w, files(corrected=True, rewrite=True)))
    rewritten = h.planner.optimize(optimization(h, parent, ctx, proof, name="rewrite"))
    assert rewritten.status == "replace" and rewritten.contract_correction["kind"] == "execution_rewrite"
    assert rewritten.changed_execution_decisions
    data = files()
    data["root.yaml"] = data["root.yaml"].replace(b"without intermediate values", b"with direct values")
    h.actions.append(lambda w, e: output(w, data))
    keep = h.planner.optimize(optimization(h, parent, name="ordinary"))
    assert keep.status == "keep" and keep.contract_correction is None


def test_review_same_context_bound_to_sealed_source_and_pipeline_failure_stays_fail(authority):
    h = authority
    ctx = context(h)
    h.actions.append(lambda w, e: response(w, files=files(corrected=True)))
    result = h.planner.design(replace(h.request, execution_context=ctx))
    assert result.status == "replace"
    def review(workspace, env):
        binding = json.loads((workspace / "inputs/review/binding.json").read_text())
        assert binding["execution_context_sha256"] == ctx.sha256
        assert (workspace / "inputs/task/execution.json").read_bytes() == ctx.as_bytes()
        assert binding["source_identity"] == "sealed-native-source"
        cite = {"source": "candidate/kernel.py", "start": 1, "end": 1, "claim": "three-stage candidate"}
        value = {"verdict": "fail", "design_identity": binding["design_identity"],
            "source_identity": binding["source_identity"], "rationale": "pipeline deviation remains",
            "obligations": [{"id": name, "verdict": "fail" if name.endswith("Tile_pipeline") else "pass",
                "reason": "inspect this fixed obligation", "evidence": [cite]} for name in binding["obligations"]],
            "substantive_change": False, "change_evidence": []}
        (workspace / "output/decision.json").write_text(json.dumps(value))
        return "reviewed"
    h.actions.append(review)
    kwargs = dict(design_ref=result.design_ref, prepared_id="prepared", operation_id="review",
        directory=h.request.directory.parent / "review", source=RevisionContent({"kernel.py": b"NUM_STAGES = 3\n"}),
        source_identity="sealed-native-source", evidence={"public_task": h.request.public_task,
            "target_hardware": "L20D", "benchmark_pass": True}, execution_context=ctx)
    review_result = h.planner.qualify(**kwargs)
    assert review_result["verdict"] == "fail" and review_result["qualified"] is False
    assert review_result["binding"]["sealed_files"] == {"kernel.py": sha256(b"NUM_STAGES = 3\n")}
    assert h.planner.qualify(**kwargs) == json.loads(json.dumps(review_result)) and len(h.calls) == 2
    with pytest.raises(ValueError, match="another prepared Task or target"):
        h.planner.qualify(**{**kwargs, "evidence": {"public_task": "another task", "target_hardware": "L20D"}})
    with pytest.raises(ValueError, match="host verified-source constructor"):
        h.planner.qualify(**{**kwargs, "execution_context": ctx.as_dict()})
    for changed_context in (None, context(h, source="configured")):
        with pytest.raises(ValueError, match="frozen Planner publication"):
            h.planner.qualify(**{**kwargs, "execution_context": changed_context})
        with pytest.raises(ValueError, match="frozen Planner publication"):
            coder_context(result.design_ref, "prepared", 2, "seed", execution_context=changed_context)
    changed_policy = context(h, policy={"framework": "triton", "sandbox_hardware": "L20D", "verify_repeats": 4})
    with pytest.raises(ValueError, match="Task or native policy"):
        authorize_contract_correction(changed_policy, result.design_ref)


def test_legacy_prompt_and_input_result_shapes_remain_exact(authority):
    h = authority
    # The baseline release's four prompt byte digests; no checkout/Git required.
    legacy_prompts = {
        "common": "ac5e48e7ee5cbd1962ede8a47372c004519df3349b7f927b43c0bb53f37732a5",
        "design": "d6d2ba65a5423e71429ce1fbe9940c11d9de168558c5a60331a0b97b50f65e6e",
        "optimize": "74df22a9698f281347073ac97395b8b8107c6afaac241cf665867f4f1f06b120",
        "reviewer": "ec06c6e15a2bebb8d0770d068c6b05b5251b36323e8a3ba484db6c94865656b5"}
    for name, digest in legacy_prompts.items():
        assert sha256(_role_prompt(name + ".md")) == digest
    parent = legacy_parent(h)
    intent = json.loads((h.request.directory / "intent.json").read_text())
    assert "task/execution.json" not in intent["inputs"]
    legacy = json.loads((h.request.directory / "result.json").read_text())
    assert "contract_correction" not in legacy
    assert PlannerResult.from_dict(legacy).design_ref == parent
    assert h.planner.design(h.request).design_ref == parent and len(h.calls) == 1
    with pytest.raises(ValueError, match="replay conflicts"):
        h.planner.design(replace(h.request, execution_context=context(h)))
    verify_release()
