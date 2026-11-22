"""Explicit, static source snapshots for AKA extraction.

Only the host selects inputs. No catalog discovery, source import, subprocess,
GPU execution or target-to-source hardware inference occurs here. Quotations
prove byte presence, not semantic truth or the authenticity of supplied claims.

Static locator and conventional comparator checks are adapted from the pure
parts of KernelSchema schema_extract/tools/{inputs,comparators}.py at
dc6e2b3d318f0324ffdb578c3dcc6cf33a64c7e8; no upstream runtime is imported.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

from aka.core.filesystem import durable_write_text, ensure_private_directory, file_signature
from .language.yaml_source import read_source_bytes
from .materials import _directory, _member

FORMAT = "aka.schema.extract-sources.v1"
ROLES = ("implementation", "host", "io", "input", "reference", "acceptance", "scope")
MAX_FILES = 256
MAX_TOTAL_BYTES = 64 * 1024 * 1024


class SourceError(ValueError):
    pass


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _loads(raw, *, exact=False):
    def unique(items):
        value = {}
        for key, item in items:
            if key in value:
                raise SourceError("duplicate source JSON member: " + key)
            value[key] = item
        return value
    try:
        return json.loads(raw, object_pairs_hook=unique, parse_float=Decimal if exact else float,
                          parse_constant=lambda value: (_ for _ in ()).throw(SourceError("nonfinite JSON")))
    except (RecursionError, ValueError) as error:
        raise SourceError("invalid bounded source JSON: " + str(error)) from error


@dataclass(frozen=True)
class SourceLocator:
    path: str | Path
    symbol: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    name: str | None = None

    def as_dict(self):
        return {**asdict(self), "path": str(self.path)}


@dataclass(frozen=True)
class SourceQuote:
    path: str
    line_start: int
    line_end: int
    quote: str
    sha256: str

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class GPUClaim:
    model: str
    kind: str
    source_digest: str
    scope_digest: str
    evidence: SourceQuote
    origin: str = "source"


@dataclass(frozen=True)
class OutputComparator:
    output: str
    source_output: str
    rule: Mapping
    evidence: tuple[SourceQuote, ...]
    # An acceptance JSON pointer binds exact parsed values mechanically. Code
    # or prose without such a pointer remains an explicit reviewed declaration.
    json_pointer: str | None = None


@dataclass(frozen=True)
class SourceDescriptor:
    sources: Mapping[str, tuple[SourceLocator, ...]]
    outputs: tuple[str, ...]
    comparators: tuple[OutputComparator, ...] = ()
    source_gpu: GPUClaim | None = None
    missing: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self):
        return {"sources": {role: [entry.as_dict() for entry in entries]
                            for role, entries in self.sources.items()},
                "outputs": list(self.outputs), "comparators": [asdict(item) for item in self.comparators],
                "source_gpu": asdict(self.source_gpu) if self.source_gpu else None,
                "missing": dict(self.missing)}

    @classmethod
    def from_dict(cls, value, *, base_directory=None):
        if set(value) - {"sources", "outputs", "comparators", "source_gpu", "missing"}:
            raise SourceError("unknown source descriptor fields")
        sources = {}
        for role, entries in value["sources"].items():
            sources[role] = []
            for entry in entries:
                item = SourceLocator(**entry)
                if base_directory is not None and not Path(item.path).is_absolute():
                    item = SourceLocator(**{**item.as_dict(), "path": str(Path(base_directory) / item.path)})
                sources[role].append(item)
            sources[role] = tuple(sources[role])
        gpu = value.get("source_gpu")
        gpu = GPUClaim(**{**gpu, "evidence": SourceQuote(**gpu["evidence"])}) if gpu else None
        comparators = tuple(OutputComparator(**{**item, "evidence": tuple(SourceQuote(**quote)
                           for quote in item["evidence"])}) for item in value.get("comparators", ()))
        return cls(sources, tuple(value["outputs"]), comparators, gpu, value.get("missing", {}))


def _lines(raw):
    # Keep newline bytes, including CRLF, for exact snippet digests/quotations.
    return raw.decode("utf-8").splitlines(keepends=True)


def _locate(locator, name, raw):
    lines = _lines(raw)
    if not lines:
        raise SourceError("source input is empty: " + name)
    first, last, symbol = 1, len(lines), locator.symbol
    if symbol is not None:
        if not isinstance(symbol, str) or not symbol.strip():
            raise SourceError("symbol must be explicit and nonempty: " + name)
        if name.endswith(".py"):
            try:
                module = ast.parse(raw.decode("utf-8"), filename=name)
                found = []
                def visit(node, parents=()):
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                            qualified = (*parents, child.name)
                            if symbol == ".".join(qualified):
                                found.append(child)
                            visit(child, qualified)
                        else:
                            visit(child, parents)
                visit(module)
            except (SyntaxError, RecursionError) as error:
                raise SourceError("cannot statically locate Python symbol: " + name) from error
            if len(found) != 1:
                raise SourceError("Python qualified symbol must resolve uniquely: " + name + "::" + symbol)
            node = found[0]
            first = min([node.lineno] + [item.lineno for item in node.decorator_list])
            last = node.end_lineno
        else:
            # Text is never parsed as executable code. A unique exact symbol
            # must occur in the explicitly selected span (or in the whole file).
            first, last = locator.start_line or first, locator.end_line or last
            span = "".join(lines[first - 1:last])
            if span.count(symbol) != 1:
                raise SourceError("text symbol must occur exactly once in its explicit span: " + name)
            if locator.start_line is None:
                first += span[:span.index(symbol)].count("\n")
                last = first + symbol.count("\n")
    if locator.start_line is not None or locator.end_line is not None:
        if (type(locator.start_line) is not int or type(locator.end_line) is not int
                or not 1 <= locator.start_line <= locator.end_line <= len(lines)):
            raise SourceError("source locator requires a valid complete line range: " + name)
        if name.endswith(".py") and symbol and (locator.start_line, locator.end_line) != (first, last):
            raise SourceError("Python symbol and explicit line range disagree: " + name)
        first, last = locator.start_line, locator.end_line
    snippet = "".join(lines[first - 1:last]).encode()
    return {"path": name, "symbol": symbol, "line_start": first, "line_end": last,
            "sha256": _digest(raw), "snippet_sha256": _digest(snippet)}


def _role_digest(roles, files, role):
    # Full files, not merely selected symbols, bind implementation versions.
    names = {entry["path"] for entry in roles[role]}
    if not names:
        return None
    return _digest(_json({"files": {name: _digest(files[name]) for name in sorted(names)},
                          "locators": roles[role]}))


def _quote(files, roles, value, *, role=None):
    item = value.as_dict() if isinstance(value, SourceQuote) else dict(value)
    required = {"path", "line_start", "line_end", "quote", "sha256"}
    if not required <= set(item):
        raise SourceError("source quote requires path, lines, exact text and full-file sha256")
    name = item["path"]
    if name not in files or item["sha256"] != _digest(files[name]):
        raise SourceError("source quote names an absent file or stale digest: " + str(name))
    first, last = item["line_start"], item["line_end"]
    lines = _lines(files[name])
    if (type(first) is not int or type(last) is not int or not 1 <= first <= last <= len(lines)
            or not isinstance(item["quote"], str) or not item["quote"].strip()):
        raise SourceError("invalid source quotation range/text")
    span = "".join(lines[first - 1:last])
    # Exact substring must start on the declared first line and end on the
    # declared last line; padding a broad range cannot cover unrelated evidence.
    at = span.find(item["quote"])
    if (at < 0 or "\n" in span[:at]
            or first + item["quote"].rstrip("\r\n").count("\n") != last):
        raise SourceError("quotation is not exact at its declared lines: " + name)
    bindings = roles.get(role, ()) if role else [entry for group in roles.values() for entry in group]
    if not any(entry["path"] == name and entry["line_start"] <= first <= last <= entry["line_end"]
               for entry in bindings):
        raise SourceError("quotation is outside the bound source role: " + str(role))
    return {key: item[key] for key in required} | {"snippet_sha256": _digest(item["quote"].encode())}


def _model(value):
    if not isinstance(value, str):
        return None
    value = re.sub(r"\s+", " ", value.strip()).casefold()
    value = re.sub(r"^nvidia\s+", "", value)
    # Never infer an actual device from SM/architecture/instruction/path names.
    if (not value or value in {"unknown", "unverified", "cuda", "gpu", "nvidia", "ada", "hopper", "blackwell"}
            or re.search(r"[/\\]|\bsm[_ -]?\d+\b|\b(compute|capability|architecture)\b", value)
            or not re.search(r"[a-z]", value) or not re.search(r"\d", value)):
        return None
    return value


def _mentions_model(text, model):
    return bool(model and isinstance(text, str) and re.search(
        r"(?<![a-z0-9])" + re.escape(model) + r"(?![a-z0-9])",
        re.sub(r"\s+", " ", text.casefold())))


def _gpu_check(claim, target, source_digest, scope_digest, files, roles):
    result = {"status": "unverified", "source": claim, "target": dict(target), "reason": ""}
    try:
        if not claim or claim.get("origin") != "source":
            raise SourceError("source GPU must be an independent source claim, never target backfill")
        if claim.get("kind") not in {"declaration", "observation"}:
            raise SourceError("source GPU requires declaration or observation provenance")
        if (not source_digest or not scope_digest or claim.get("source_digest") != source_digest
                or claim.get("scope_digest") != scope_digest):
            raise SourceError("source GPU claim is not bound to this implementation version and scope")
        quote = _quote(files, roles, claim["evidence"])
        source_model = _model(claim.get("model"))
        if not _mentions_model(quote["quote"], source_model):
            raise SourceError("source GPU model is absent from its exact source evidence")
        target_model = _model(target.get("model"))
        if (not target_model or target.get("kind") not in {"declaration", "observation"}
                or target.get("origin") != "environment" or not isinstance(target.get("evidence"), str)
                or target.get("source", "explicit") in {"unknown", "unverified"}
                or not _mentions_model(target["evidence"], target_model)):
            raise SourceError("target GPU needs a separately bound Environment declaration/observation")
        result["status"] = "matched" if source_model == target_model else "mismatch"
        result["reason"] = ("same explicit device model/variant; provenance kinds preserved"
                            if result["status"] == "matched" else "source and target device model/variant differ")
    except (SourceError, KeyError, TypeError) as error:
        result["reason"] = str(error)
    return result


def validate_target_gpu(target_gpu: Mapping, native_hardware: str):
    """Bind the independent Environment fact to the actual native selection."""
    model = _model(target_gpu.get("model"))
    if (not model or model != _model(native_hardware)
            or target_gpu.get("origin") != "environment"
            or target_gpu.get("kind") not in {"declaration", "observation"}
            or not isinstance(target_gpu.get("evidence"), str)
            or target_gpu.get("source", "explicit") in {"unknown", "unverified"}
            or not _mentions_model(target_gpu["evidence"], model)):
        raise SourceError("target GPU must match the independent native Environment hardware selection")
    return dict(target_gpu)


def _pointer(value, pointer):
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise SourceError("acceptance JSON pointer must be explicit")
    for part in pointer.split("/")[1:] if pointer else ():
        part = part.replace("~1", "/").replace("~0", "~")
        try:
            value = value[int(part)] if isinstance(value, list) else value[part]
        except (KeyError, IndexError, ValueError, TypeError) as error:
            raise SourceError("acceptance JSON pointer does not resolve") from error
    return value


def _scalar(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float, Decimal)):
        return str(value)
    if isinstance(value, str):
        return value
    raise SourceError("comparison rules must contain explicit scalar values")


def _same_scalar(left, right):
    if type(left) is bool or type(right) is bool or left is None or right is None:
        return type(left) is type(right) and left == right
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except InvalidOperation:
        return left == right


def _comparators(descriptor, files, roles):
    result, seen = [], set()
    for entry in descriptor.comparators:
        if entry.output not in descriptor.outputs or entry.output in seen or not entry.source_output:
            raise SourceError("comparator needs a unique, declared output and explicit source_output")
        seen.add(entry.output)
        if entry.source_output in descriptor.outputs and entry.source_output != entry.output:
            raise SourceError("an identically named source output must bind its own criterion")
        if not entry.rule or not entry.evidence:
            raise SourceError("comparator requires a nonempty rule and source evidence")
        for key, value in entry.rule.items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key):
                raise SourceError("invalid comparison rule key")
            _scalar(value)
            if key in {"atol", "rtol", "max_rel_l2", "max_mismatch_fraction"} and value is not None:
                try:
                    number = Decimal(str(value))
                    if not number.is_finite() or number < 0:
                        raise InvalidOperation
                except InvalidOperation as error:
                    raise SourceError("comparison threshold must be finite and nonnegative") from error
            if key == "relative_to" and value not in {"reference", "candidate"}:
                raise SourceError("relative_to must explicitly name reference or candidate")
        quotes = [_quote(files, roles, quote, role="acceptance") for quote in entry.evidence]
        status = "declared"
        if entry.json_pointer is not None:
            paths = {quote["path"] for quote in quotes}
            if len(paths) != 1:
                raise SourceError("a structured comparator must bind one acceptance JSON file")
            name = next(iter(paths))
            document = _loads(files[name], exact=True)
            declared = _pointer(document, entry.json_pointer)
            parts = [part.replace("~1", "/").replace("~0", "~")
                     for part in entry.json_pointer.split("/")[1:]]
            if "outputs" in parts:
                at = parts.index("outputs") + 1
                if at >= len(parts) or parts[at] != entry.source_output:
                    raise SourceError("acceptance JSON pointer changes the declared source output")
            if (not isinstance(declared, dict) or set(declared) != set(entry.rule)
                    or any(not _same_scalar(value, declared[key]) for key, value in entry.rule.items())):
                raise SourceError("comparator differs from its exact acceptance JSON object")
            # Its actual values, not only an unrelated JSON line, must be quoted.
            joined = "\n".join(quote["quote"] for quote in quotes)
            for key, value in declared.items():
                if not any(_same_scalar(token, _scalar(value)) for token in _assignments(joined, key)):
                    raise SourceError("comparator evidence omits its declaration: " + key)
            status = "structured_declaration"
        else:
            # No inferred formula: every claimed scalar must occur as an
            # explicit key=value/key:value declaration in quoted source text.
            joined = "\n".join(quote["quote"] for quote in quotes)
            for key, value in entry.rule.items():
                if not _assignments(joined, key) or not any(_same_scalar(token, _scalar(value))
                                                         for token in _assignments(joined, key)):
                    raise SourceError("unanchored comparator declaration: " + key)
        result.append({"output": entry.output, "source_output": entry.source_output,
                       "rule": {key: str(value) if isinstance(value, Decimal) else value
                                for key, value in entry.rule.items()}, "evidence": quotes, "status": status,
                       "json_pointer": entry.json_pointer})
    for output in descriptor.outputs:
        if output not in seen:
            result.append({"output": output, "source_output": None, "rule": None,
                           "evidence": [], "status": "unavailable", "json_pointer": None})
    return result


def inspect_sources(descriptor: SourceDescriptor, *, target_gpu: Mapping | None = None):
    """Read only explicit files; return (manifest, raw-files) without importing them.

    First inspection may omit the GPU claim to obtain source/scope digests for
    an explicit host declaration. This never derives the source GPU from target.
    """
    if set(descriptor.sources) - set(ROLES) or set(descriptor.missing) - set(ROLES):
        raise SourceError("unknown source role")
    if (not descriptor.outputs or len(set(descriptor.outputs)) != len(descriptor.outputs)
            or any(not isinstance(name, str) or not name.strip() for name in descriptor.outputs)):
        raise SourceError("source descriptor requires unique explicit output names")
    files, paths, signatures, roles, missing = {}, {}, {}, {}, {}
    for role in ROLES:
        roles[role] = []
        for locator in descriptor.sources.get(role, ()):
            path = _directory(locator.path)
            name = locator.name or path.name
            _member(name)
            if name == "manifest.json":
                raise SourceError("manifest.json is a reserved source name")
            if name in paths and paths[name] != path:
                raise SourceError("source logical name maps to multiple paths: " + name)
            if name not in files:
                if len(files) >= MAX_FILES:
                    raise SourceError("too many source files")
                signatures[name] = file_signature(path.lstat())
                raw = read_source_bytes(path)
                raw.decode("utf-8")
                files[name], paths[name] = raw, path
                if sum(map(len, files.values())) > MAX_TOTAL_BYTES:
                    raise SourceError("source snapshot exceeds total byte limit")
            roles[role].append(_locate(locator, name, files[name]))
        if not roles[role]:
            missing[role] = descriptor.missing.get(role, "not supplied")
            if not isinstance(missing[role], str) or not missing[role].strip():
                raise SourceError("missing source role needs a nonempty diagnostic")
        elif role in descriptor.missing:
            raise SourceError("source role is both supplied and declared missing: " + role)
    if not roles["implementation"]:
        raise SourceError("source implementation is required")
    for reference in roles["reference"]:
        for implementation in roles["implementation"]:
            if (reference["path"] == implementation["path"]
                    and max(reference["line_start"], implementation["line_start"])
                    <= min(reference["line_end"], implementation["line_end"])):
                raise SourceError("reference must be independent of the selected implementation span")
    for name, path in paths.items():
        if file_signature(path.lstat()) != signatures[name]:
            raise SourceError("source changed during snapshot: " + name)
    source_digest = _role_digest(roles, files, "implementation")
    scope_digest = _role_digest(roles, files, "scope")
    gpu = asdict(descriptor.source_gpu) if descriptor.source_gpu else None
    comparator_rows = _comparators(descriptor, files, roles)
    manifest = {"format": FORMAT, "files": {name: {"sha256": _digest(raw), "bytes": len(raw)}
                                           for name, raw in sorted(files.items())},
                "roles": roles, "missing": missing, "outputs": list(descriptor.outputs),
                "source_digest": source_digest, "scope_digest": scope_digest,
                "request_digest": _request_digest(descriptor, target_gpu or {}),
                "comparators": comparator_rows,
                "gpu_check": _gpu_check(gpu, target_gpu or {}, source_digest, scope_digest, files, roles),
                "diagnostics": [{"category": "source.missing", "role": role, "message": reason}
                                for role, reason in missing.items()] + [
                    {"category": "comparison.unavailable", "output": row["output"],
                     "message": "no source comparator supplied; no defaults inferred"}
                    for row in comparator_rows if row["status"] == "unavailable"],
                "limitations": "Exact quotations and structured declarations are evidence, not semantic proof; "
                                "source acceptance is separate from the native target Task acceptance."}
    return manifest, files


def _request_digest(descriptor, target_gpu):
    return _digest(_json({"descriptor": descriptor.as_dict(), "target_gpu": dict(target_gpu)}))


@dataclass(frozen=True)
class SourceSnapshot:
    directory: Path
    identity: str
    manifest: dict
    # This is the complete read-only role mount view, including its manifest.
    files: dict[str, bytes]

    def ref(self):
        return {"directory": str(self.directory), "identity": self.identity}

    def require_same_gpu(self):
        check = self.manifest["gpu_check"]
        if check["status"] != "matched":
            raise SourceError("source GPU " + check["status"] + ": " + check["reason"])
        return check

    def verify_quote(self, quote, *, role=None):
        files = {name: self.files["sources/" + name] for name in self.manifest["files"]}
        return _quote(files, self.manifest["roles"], quote, role=role)

    def source_reference(self, role, index=0):
        row = self.manifest["roles"][role][index]
        suffix = "::" + row["symbol"] if row["symbol"] else (
            f"#L{row['line_start']}-L{row['line_end']}")
        return f"aka-source@{row['sha256']}:{row['path']}{suffix}"


def snapshot_sources(directory, descriptor: SourceDescriptor, *, target_gpu: Mapping):
    """Publish manifest last; a committed snapshot survives original-source loss.

    Caller owns the operation lock. Existing committed data is checked against
    the frozen request, without reopening original files. The operation owner
    should retain ref() and use read_source_snapshot(ref) on later recovery.
    An incomplete directory cannot be silently overwritten.
    """
    directory = _directory(directory)
    if directory.exists():
        manifest = _loads(read_source_bytes(directory / "manifest.json"))
        previous = read_source_snapshot({"directory": str(directory), "identity": _digest(_json(manifest))})
        return validate_source_request(previous, descriptor, target_gpu)
    manifest, files = inspect_sources(descriptor, target_gpu=target_gpu)
    identity = _digest(_json(manifest))
    ensure_private_directory(directory)
    for name, raw in files.items():
        destination = directory / "files" / _member(name)
        ensure_private_directory(destination.parent)
        durable_write_text(destination, raw.decode("utf-8"), mode=0o444)
    durable_write_text(directory / "manifest.json", _json(manifest).decode(), mode=0o444)
    return read_source_snapshot({"directory": str(directory), "identity": identity})


def read_source_snapshot(ref):
    if isinstance(ref, SourceSnapshot):
        ref = ref.ref()
    if not isinstance(ref, Mapping) or set(ref) != {"directory", "identity"}:
        raise SourceError("source snapshot read requires its directory and fixed identity")
    directory = _directory(ref["directory"])
    manifest_raw = read_source_bytes(directory / "manifest.json")
    manifest = _loads(manifest_raw)
    if manifest.get("format") != FORMAT or _digest(_json(manifest)) != ref["identity"]:
        raise SourceError("source snapshot manifest identity mismatch")
    files = {}
    if len(manifest["files"]) > MAX_FILES:
        raise SourceError("too many retained source files")
    for name, expected in manifest["files"].items():
        raw = read_source_bytes(_directory(directory / "files" / _member(name)))
        if _digest(raw) != expected["sha256"] or len(raw) != expected["bytes"]:
            raise SourceError("source snapshot byte mismatch: " + name)
        files[name] = raw
        if sum(map(len, files.values())) > MAX_TOTAL_BYTES:
            raise SourceError("retained source snapshot exceeds total byte limit")
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    if actual != {"manifest.json", *("files/" + name for name in files)}:
        raise SourceError("source snapshot inventory mismatch")
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise SourceError("source snapshot contains a link")
    for role, rows in manifest["roles"].items():
        if role not in ROLES:
            raise SourceError("unknown retained source role")
        for row in rows:
            located = _locate(SourceLocator(row["path"], row["symbol"], row["line_start"], row["line_end"]),
                              row["path"], files[row["path"]])
            if located != row:
                raise SourceError("source locator no longer matches its bytes")
    for role, key in (("implementation", "source_digest"), ("scope", "scope_digest")):
        if _role_digest(manifest["roles"], files, role) != manifest[key]:
            raise SourceError("source role digest mismatch")
    check = manifest["gpu_check"]
    if _gpu_check(check["source"], check["target"], manifest["source_digest"], manifest["scope_digest"],
                  files, manifest["roles"]) != check:
        raise SourceError("GPU check is inconsistent with retained evidence")
    return SourceSnapshot(directory, ref["identity"], manifest,
                          {"sources/" + name: raw for name, raw in files.items()} |
                          {"sources/manifest.json": manifest_raw})


def validate_source_reference(snapshot: SourceSnapshot, value: str, *, role: str):
    """Bind a Schema source string to the frozen, explicitly selected locator."""
    if role not in ROLES or not isinstance(value, str):
        raise SourceError("invalid source reference role or text")
    for index, row in enumerate(snapshot.manifest["roles"][role]):
        expected = snapshot.source_reference(role, index)
        if value == expected:
            return dict(row)
    raise SourceError("source reference must identify a fixed selected " + role + " locator")


def validate_source_request(snapshot: SourceSnapshot, descriptor: SourceDescriptor, target_gpu: Mapping):
    """Check replay's frozen request without consulting vanished upstream files."""
    if snapshot.manifest["request_digest"] != _request_digest(descriptor, target_gpu):
        raise SourceError("source snapshot request differs from its frozen input")
    return snapshot


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
COMPARATOR_FIELDS = {"atol", "rtol", "relative_to", "mode", "max_rel_l2",
                     "max_mismatch_fraction", "equal_nan", "comparison_dtype"}
FORMULA = re.compile(r"abs\(\s*(candidate|reference)\s*-\s*(candidate|reference)\s*\)\s*<=\s*"
                     rf"(?:({NUMBER}|atol)\s*\+\s*)?({NUMBER}|rtol)\s*\*\s*abs\(\s*(candidate|reference)\s*\)")


def _assignments(text, key):
    return re.findall(r'\b' + re.escape(key) + r'''["']?\s*[:=]\s*["']?([A-Za-z0-9_.+\-]+)''', text)


def validate_comparisons(snapshot: SourceSnapshot, criteria: Mapping[str, str], bindings=None):
    """P1 hard gate for per-output source rules; missing rules stay unknown.

    Arbitrary mathematical prose still needs independent review. Supported
    conventional elementwise formulas additionally check numeric operands.
    This does not replace the native target Task comparator.
    """
    if set(criteria) != set(snapshot.manifest["outputs"]):
        raise SourceError("Contract comparator outputs differ from the explicit source outputs")
    rows = {row["output"]: row for row in snapshot.manifest["comparators"]}
    if bindings is not None:
        if (len(bindings) != len(rows) or {item.get("output") for item in bindings} != set(rows)):
            raise SourceError("comparison review bindings must cover every output exactly once")
        for item in bindings:
            row = rows[item["output"]]
            if item.get("source_output") != row["source_output"]:
                raise SourceError("comparison review changes source-output identity")
            if row["rule"]:
                quotes = [snapshot.verify_quote(quote, role="acceptance") for quote in item.get("evidence", ())]
                if not quotes or not all(any(quote["path"] == expected["path"] and
                       quote["line_start"] <= expected["line_start"] <= expected["line_end"] <= quote["line_end"]
                       and expected["quote"] in quote["quote"] for quote in quotes) for expected in row["evidence"]):
                    raise SourceError("comparison review must cite its actual declaration")
    for output, text in criteria.items():
        if not isinstance(text, str):
            raise SourceError("Contract comparison criterion must be explicit text")
        rule = rows[output]["rule"]
        if not rule:
            if (not re.search(r"\b(unknown|unavailable|not supplied)\b", text, re.I)
                    or re.search(r"\b(atol|rtol|relative_to|max_rel_l2|max_mismatch_fraction)\s*[:=]", text)
                    or FORMULA.search(text)):
                raise SourceError("unavailable source comparator must remain explicitly unknown: " + output)
            continue
        for key, value in rule.items():
            tokens = _assignments(text, key)
            if not tokens or any(not _same_scalar(token, _scalar(value)) for token in tokens):
                raise SourceError("Contract changes or omits source comparator " + output + "." + key)
        if any(_assignments(text, key) for key in COMPARATOR_FIELDS - set(rule)):
            raise SourceError("Contract invents an undeclared comparator parameter: " + output)
        for match in FORMULA.finditer(text):
            left, right, atol, rtol, basis = match.groups()
            if left == right or basis != rule.get("relative_to"):
                raise SourceError("comparison formula contradicts its operands: " + output)
            for key, token in (("atol", atol or "0"), ("rtol", rtol)):
                if token != key and key in rule and not _same_scalar(token, rule[key]):
                    raise SourceError("comparison formula changes source threshold: " + output + "." + key)
    return tuple(row for row in snapshot.manifest["comparators"] if row["status"] == "unavailable")
