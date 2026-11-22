"""CPU-only source evidence checks; all GPU claims below are synthetic fixtures."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from aka.workflow.schema.extract_sources import (
    GPUClaim, OutputComparator, SourceDescriptor, SourceError, SourceLocator, SourceQuote,
    inspect_sources, read_source_snapshot, snapshot_sources, validate_comparisons,
    validate_source_reference, validate_target_gpu,
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def quote(path, first=1, last=None):
    raw = path.read_bytes()
    lines = raw.decode().splitlines(keepends=True)
    last = last or len(lines)
    return SourceQuote(path.name, first, last, "".join(lines[first - 1:last]), sha(raw))


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "original"
    root.mkdir()
    data = {
        "kernel.py": "raise RuntimeError('SOURCE MUST NEVER EXECUTE')\n\nclass Model:\n    def forward(self, x):\n        return x\n",
        "reference.py": "def reference(x):\n    return x\n",
        "input.py": "def make_inputs(n):\n    return {'x': n}\n",
        "scope.md": "Synthetic fixture; not a real GPU measurement.\nDevice: NVIDIA L20D\nScope: x length 1024, output y.\n",
        "acceptance.json": '{"outputs":{"y":{"atol":0.01,"rtol":0.05,"relative_to":"candidate","mode":"elementwise"}}}\n',
    }
    for name, text in data.items():
        (root / name).write_text(text)
    loc = lambda name, symbol=None: SourceLocator(root / name, symbol=symbol)
    rule = {"atol": "0.01", "rtol": "0.05", "relative_to": "candidate", "mode": "elementwise"}
    comparator = OutputComparator("y", "y", rule, (quote(root / "acceptance.json"),), "/outputs/y")
    descriptor = SourceDescriptor({
        "implementation": (loc("kernel.py", "Model.forward"),),
        "host": (loc("kernel.py", "Model"),),
        "io": (loc("scope.md"),), "input": (loc("input.py", "make_inputs"),),
        "reference": (loc("reference.py", "reference"),),
        "acceptance": (loc("acceptance.json"),), "scope": (loc("scope.md"),),
    }, ("y",), (comparator,))
    manifest, _ = inspect_sources(descriptor)
    descriptor = replace(descriptor, source_gpu=GPUClaim("NVIDIA L20D", "declaration",
        manifest["source_digest"], manifest["scope_digest"], quote(root / "scope.md", 2, 2)))
    return root, descriptor


TARGET = {"model": "NVIDIA L20D", "kind": "declaration", "origin": "environment",
          "evidence": "Synthetic Environment fixture explicitly requests NVIDIA L20D"}
CRITERIA = {"y": "atol=0.01, rtol=0.05, relative_to=candidate, mode=elementwise; "
                 "abs(candidate-reference) <= 0.01 + 0.05 * abs(candidate)"}


def frozen(tmp_path, descriptor, target=TARGET):
    return snapshot_sources(tmp_path / "snapshot", descriptor, target_gpu=target)


def test_static_snapshot_roundtrip_original_can_disappear(source, tmp_path):
    root, descriptor = source
    snapshot = frozen(tmp_path, descriptor)
    assert snapshot.require_same_gpu()["source"]["kind"] == "declaration"
    assert snapshot.manifest["gpu_check"]["target"]["kind"] == "declaration"
    assert set(snapshot.files) == {"sources/" + p.name for p in root.iterdir()} | {"sources/manifest.json"}
    assert snapshot.manifest["missing"] == {}
    assert validate_comparisons(snapshot, CRITERIA) == ()
    for path in root.iterdir():
        path.unlink()
    assert read_source_snapshot(snapshot.ref()).identity == snapshot.identity
    assert frozen(tmp_path, descriptor).identity == snapshot.identity


def test_descriptor_serialization_and_idempotent_snapshot(source, tmp_path):
    _, descriptor = source
    roundtrip = SourceDescriptor.from_dict(json.loads(json.dumps(descriptor.as_dict())))
    assert roundtrip.as_dict() == descriptor.as_dict()
    first = frozen(tmp_path, descriptor)
    assert frozen(tmp_path, roundtrip).ref() == first.ref()


def test_source_reference_requires_actual_role_digest_and_symbol(source, tmp_path):
    _, descriptor = source
    snapshot = frozen(tmp_path, descriptor)
    value = snapshot.source_reference("implementation")
    assert validate_source_reference(snapshot, value, role="implementation")["symbol"] == "Model.forward"
    for bad in ("kernel.py::Model.forward", value.replace("Model.forward", "Model"),
                value.replace("aka-source@", "aka-source@0")):
        with pytest.raises(SourceError):
            validate_source_reference(snapshot, bad, role="implementation")
    with pytest.raises(SourceError):
        validate_source_reference(snapshot, value, role="reference")


@pytest.mark.parametrize("claim", ["missing", "stale_source", "stale_scope", "target", "unknown", "sm", "wrong_quote"])
def test_unverified_source_gpu_never_passes(source, tmp_path, claim):
    _, descriptor = source
    gpu = descriptor.source_gpu
    if claim == "missing":
        gpu = None
    elif claim == "stale_source":
        gpu = replace(gpu, source_digest="0" * 64)
    elif claim == "stale_scope":
        gpu = replace(gpu, scope_digest="0" * 64)
    elif claim == "target":
        gpu = replace(gpu, origin="target")
    elif claim == "unknown":
        gpu = replace(gpu, kind="unknown")
    elif claim == "sm":
        gpu = replace(gpu, model="SM89")
    else:
        gpu = replace(gpu, evidence=replace(gpu.evidence, quote="NVIDIA H100"))
    snapshot = frozen(tmp_path, replace(descriptor, source_gpu=gpu))
    assert snapshot.manifest["gpu_check"]["status"] == "unverified"
    with pytest.raises(SourceError):
        snapshot.require_same_gpu()


@pytest.mark.parametrize("target", [
    {**TARGET, "model": "L20", "evidence": "Environment L20"},
    {**TARGET, "model": "H100", "evidence": "Environment H100"},
    {**TARGET, "model": "L20D", "origin": "source"},
    {**TARGET, "kind": "unknown"}, {**TARGET, "evidence": "SM89"},
    {**TARGET, "source": "unknown"},
])
def test_independent_exact_target_is_required(source, tmp_path, target):
    _, descriptor = source
    with pytest.raises(SourceError):
        frozen(tmp_path, descriptor, target).require_same_gpu()


def test_observation_and_declaration_remain_distinct(source, tmp_path):
    _, descriptor = source
    snapshot = frozen(tmp_path, replace(descriptor, source_gpu=replace(descriptor.source_gpu, kind="observation")))
    check = snapshot.require_same_gpu()
    assert check["source"]["kind"] == "observation"
    assert check["target"]["kind"] == "declaration"


def test_changed_code_and_scope_cannot_reuse_gpu_claim(source, tmp_path):
    root, descriptor = source
    (root / "kernel.py").write_text((root / "kernel.py").read_text() + "# modified\n")
    with pytest.raises(SourceError, match="version"):
        frozen(tmp_path, descriptor).require_same_gpu()


def test_missing_roles_and_comparator_are_diagnostic_not_defaulted(source, tmp_path):
    _, descriptor = source
    descriptor = replace(descriptor, sources={k: v for k, v in descriptor.sources.items()
                         if k not in {"acceptance", "reference", "input"}}, comparators=(),
                         missing={"acceptance": "upstream comparator unavailable"})
    snapshot = frozen(tmp_path, descriptor)
    assert snapshot.manifest["missing"]["acceptance"] == "upstream comparator unavailable"
    assert snapshot.manifest["comparators"][0]["rule"] is None
    assert len(validate_comparisons(snapshot, {"y": "unknown; source comparator unavailable"})) == 1
    with pytest.raises(SourceError, match="unknown"):
        validate_comparisons(snapshot, {"y": "unknown; atol=0.01"})


@pytest.mark.parametrize("change", [
    {"y": CRITERIA["y"].replace("relative_to=candidate", "relative_to=reference")},
    {"y": CRITERIA["y"].replace("abs(candidate)", "abs(reference)")},
    {"y": CRITERIA["y"].replace("<= 0.01", "<= 0.02")},
    {"y": CRITERIA["y"] + "; rtol=0.1"},
    {"z": CRITERIA["y"]}, {"y": "torch.allclose with ordinary tolerances"},
])
def test_comparator_wrong_operands_thresholds_or_outputs_rejected(source, tmp_path, change):
    _, descriptor = source
    with pytest.raises(SourceError):
        validate_comparisons(frozen(tmp_path, descriptor), change)


def test_comparator_must_match_actual_structured_declaration(source):
    _, descriptor = source
    entry = descriptor.comparators[0]
    with pytest.raises(SourceError, match="exact acceptance"):
        inspect_sources(replace(descriptor, comparators=(replace(entry, rule={**entry.rule, "rtol": "0.1"}),)))


def test_acceptance_exact_decimal_no_float_collapse(source):
    root, descriptor = source
    path = root / "acceptance.json"
    path.write_text('{"outputs":{"y":{"rtol":0.10000000000000000000001}}}\n')
    entry = OutputComparator("y", "y", {"rtol": "0.1"}, (quote(path),), "/outputs/y")
    with pytest.raises(SourceError, match="exact acceptance"):
        inspect_sources(replace(descriptor, comparators=(entry,)))
    entry = replace(entry, rule={"rtol": "0.10000000000000000000001"})
    manifest, _ = inspect_sources(replace(descriptor, comparators=(entry,)))
    assert manifest["comparators"][0]["rule"]["rtol"].endswith("001")


def test_review_comparison_quotes_cover_real_rules(source, tmp_path):
    _, descriptor = source
    snapshot = frozen(tmp_path, descriptor)
    evidence = descriptor.comparators[0].evidence[0].as_dict()
    bindings = [{"output": "y", "source_output": "y", "evidence": [evidence]}]
    validate_comparisons(snapshot, CRITERIA, bindings)
    bindings[0]["evidence"] = [descriptor.source_gpu.evidence.as_dict()]
    with pytest.raises(SourceError):
        validate_comparisons(snapshot, CRITERIA, bindings)


def test_exact_quote_rejects_wrong_span_digest_or_text(source, tmp_path):
    root, descriptor = source
    snapshot = frozen(tmp_path, descriptor)
    correct = quote(root / "scope.md", 2, 2)
    assert snapshot.verify_quote(correct)["snippet_sha256"] == sha(correct.quote.encode())
    for bad in (replace(correct, line_start=1), replace(correct, line_end=3),
                replace(correct, sha256="0" * 64), replace(correct, quote="Device: NVIDIA L20\n")):
        with pytest.raises(SourceError):
            snapshot.verify_quote(bad)


def test_reference_overlap_and_ambiguous_python_symbol_rejected(source):
    root, descriptor = source
    roles = dict(descriptor.sources)
    roles["reference"] = roles["implementation"]
    with pytest.raises(SourceError, match="independent"):
        inspect_sources(replace(descriptor, sources=roles))
    (root / "kernel.py").write_text("def forward(x): return x\ndef forward(x): return x\n")
    roles = {**descriptor.sources, "implementation": (SourceLocator(root / "kernel.py", "forward"),)}
    with pytest.raises(SourceError, match="uniquely"):
        inspect_sources(replace(descriptor, sources=roles))


def test_explicit_text_symbol_is_unique_and_span_bound(tmp_path):
    path = tmp_path / "kernel.cu"
    path.write_text("// never compile this\nvoid actual_kernel() { }\nvoid other() { }\n")
    descriptor = SourceDescriptor({"implementation": (SourceLocator(path, "actual_kernel", 2, 2),)}, ("out",))
    manifest, _ = inspect_sources(descriptor)
    assert manifest["roles"]["implementation"][0]["line_start"] == 2
    with pytest.raises(SourceError):
        inspect_sources(replace(descriptor, sources={"implementation": (SourceLocator(path, "void"),)}))


def test_snapshot_byte_tamper_extra_file_and_identity_rejected(source, tmp_path):
    _, descriptor = source
    snapshot = frozen(tmp_path, descriptor)
    with pytest.raises(SourceError, match="identity"):
        read_source_snapshot({**snapshot.ref(), "identity": "0" * 64})
    extra = snapshot.directory / "extra"
    extra.write_text("unlisted")
    with pytest.raises(SourceError, match="inventory"):
        read_source_snapshot(snapshot.ref())
    extra.unlink()
    path = snapshot.directory / "files/kernel.py"
    path.chmod(0o600)
    path.write_text("changed")
    with pytest.raises(SourceError, match="byte mismatch"):
        read_source_snapshot(snapshot.ref())


def test_source_symlink_and_name_collision_rejected(source, tmp_path):
    root, descriptor = source
    link = root / "alias.py"
    link.symlink_to(root / "kernel.py")
    with pytest.raises(ValueError, match="links"):
        inspect_sources(replace(descriptor, sources={"implementation": (SourceLocator(link),)}))
    roles = {**descriptor.sources, "host": (SourceLocator(root / "reference.py", name="kernel.py"),)}
    with pytest.raises(SourceError, match="multiple paths"):
        inspect_sources(replace(descriptor, sources=roles))


def test_scope_locator_change_changes_binding(source):
    _, descriptor = source
    before, _ = inspect_sources(descriptor)
    locator = descriptor.sources["scope"][0]
    changed = replace(descriptor, sources={**descriptor.sources,
        "scope": (replace(locator, start_line=2, end_line=3),)})
    after, _ = inspect_sources(changed)
    assert before["scope_digest"] != after["scope_digest"]
    assert after["gpu_check"]["status"] == "unverified"


def test_target_mapping_must_match_native_selected_device():
    assert validate_target_gpu(TARGET, "L20D") == TARGET
    for name in ("H100", "L20", "SM89", "unknown", "nvidia"):
        with pytest.raises(SourceError):
            validate_target_gpu(TARGET, name)


def test_snapshot_replay_rejects_changed_request(source, tmp_path):
    _, descriptor = source
    frozen(tmp_path, descriptor)
    with pytest.raises(SourceError, match="request differs"):
        frozen(tmp_path, descriptor, {**TARGET, "kind": "observation"})


def test_gpu_variant_is_not_a_substring_alias(source, tmp_path):
    _, descriptor = source
    # The real quoted fixture is L20D, not L20. Both requested labels cannot
    # make this a valid source declaration merely by agreeing with each other.
    descriptor = replace(descriptor, source_gpu=replace(descriptor.source_gpu, model="L20"))
    with pytest.raises(SourceError):
        frozen(tmp_path, descriptor, {**TARGET, "model": "L20", "evidence": "L20 requested"}).require_same_gpu()


def test_undeclared_comparator_defaults_are_not_invented(source, tmp_path):
    _, descriptor = source
    snapshot = frozen(tmp_path, descriptor)
    with pytest.raises(SourceError, match="invents"):
        validate_comparisons(snapshot, {"y": CRITERIA["y"] + "; max_rel_l2=0.1"})


def test_structured_comparator_binds_source_output_and_nullable_fields(source):
    root, descriptor = source
    path = root / "acceptance.json"
    path.write_text('{"outputs":{"z":{"rtol":0.1,"max_rel_l2":null,"equal_nan":false}}}\n')
    entry = OutputComparator("y", "y", {"rtol": "0.1", "max_rel_l2": None, "equal_nan": False},
                             (quote(path),), "/outputs/z")
    with pytest.raises(SourceError, match="source output"):
        inspect_sources(replace(descriptor, comparators=(entry,)))
    entry = replace(entry, source_output="z")
    manifest, _ = inspect_sources(replace(descriptor, comparators=(entry,)))
    assert manifest["comparators"][0]["rule"]["max_rel_l2"] is None
