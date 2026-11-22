"""Explicit public selection of AKA's own Schema driver for native Light."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Mapping

FORMAT = "aka.schema.launch.v1"


def _json_object(path):
    from aka.core.filesystem import read_regular_bytes
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate Schema configuration field: " + key)
            result[key] = value
        return result
    value = json.loads(read_regular_bytes(path, limit=8 * 1024 * 1024), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite configuration")))
    if not isinstance(value, dict):
        raise ValueError("Schema configuration must be a JSON object")
    return value


def _path(value, base):
    path = Path(value).expanduser()
    return str((base / path if not path.is_absolute() else path).resolve(strict=True))


@dataclass(frozen=True)
class SchemaConfig:
    continuous: bool = False
    limits: Mapping = field(default_factory=dict)
    initial: Mapping | None = None
    sota: Mapping | None = None
    wiki_ids: tuple[str, ...] = ()
    planner: Mapping = field(default_factory=dict)

    @classmethod
    def load(cls, value=None, *, initial_schema=None):
        if isinstance(value, cls):
            value = value.as_dict()
        base = Path.cwd()
        if isinstance(value, (str, Path)):
            path = Path(value).expanduser().resolve(strict=True)
            base = path.parent
            value = _json_object(path)
            if value.get("format") != FORMAT:
                raise ValueError("Retired strict/dynamic Schema configuration; use format " + FORMAT)
        value = dict(value or {})
        if value.pop("format", FORMAT) != FORMAT:
            raise ValueError("Retired strict/dynamic Schema configuration; use format " + FORMAT)
        allowed = {"continuous", "limits", "initial", "sota", "wiki_ids", "planner"}
        if set(value) - allowed:
            raise ValueError("Unsupported or retired Schema configuration fields: " + ", ".join(sorted(set(value) - allowed)))
        if type(value.get("continuous", False)) is not bool:
            raise ValueError("Schema continuous must be boolean")
        for key in ("limits", "planner"):
            if not isinstance(value.get(key, {}), Mapping):
                raise ValueError(key + " must be an object")
        if "continuous" in value.get("limits", {}):
            raise ValueError("Select continuous at the Schema configuration root")
        wiki = value.get("wiki_ids", ())
        if not isinstance(wiki, (list, tuple)) or any(not isinstance(item, str) for item in wiki):
            raise ValueError("wiki_ids must be a list of fixed material IDs")
        value["wiki_ids"] = tuple(wiki)
        initial = value.get("initial")
        if initial_schema is not None:
            if not isinstance(initial, Mapping):
                raise ValueError("--schema-path requires initial source_files, hardware and evidence in --schema-config")
            initial = {**initial, "schema": _path(initial_schema, Path.cwd())}
        if initial is not None:
            if not isinstance(initial, Mapping) or set(initial) != {"schema", "source_files", "hardware", "evidence"}:
                raise ValueError("initial requires schema, full source_files SHA-256 map, hardware and evidence paths")
            initial = dict(initial)
            initial["schema"] = _path(initial["schema"], base)
            hashes = initial["source_files"]
            if (not isinstance(hashes, Mapping) or not hashes or any(
                    not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
                    or not isinstance(sha, str) or len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha)
                    for name, sha in hashes.items())):
                raise ValueError("initial.source_files must bind every native source file to a lowercase SHA-256")
            if not isinstance(initial["hardware"], str) or not initial["hardware"].strip():
                raise ValueError("initial.hardware must identify the native target")
            if not isinstance(initial["evidence"], (list, tuple)) or not initial["evidence"]:
                raise ValueError("initial.evidence requires pairing provenance paths")
            initial["evidence"] = [_path(item, base) for item in initial["evidence"]]
            value["initial"] = initial
        sota = value.get("sota")
        if sota is not None:
            if initial is not None:
                raise ValueError("initial kernel/Schema and source-only sota are mutually exclusive")
            if not isinstance(sota, Mapping) or set(sota) != {"descriptor", "target_gpu", "target_hardware"}:
                raise ValueError("sota requires descriptor, target_gpu and target_hardware")
            from .extract_sources import SourceDescriptor
            descriptor = SourceDescriptor.from_dict(sota["descriptor"], base_directory=base)
            value["sota"] = {**sota, "descriptor": descriptor.as_dict()}
        planner = dict(value.get("planner", {}))
        if "agent_cli" in planner:
            raise ValueError("Schema Planner borrows settings.agent_cli; do not select another backend")
        from .language import CheckPolicy
        from .planner import PlannerConfig
        policy = CheckPolicy(**planner.get("check_policy", {}))
        PlannerConfig(**{**planner, "check_policy": policy})
        # Validate fields/types now; a native finite max_iters supplies the root
        # episode allowance later when no explicit root total is selected.
        from .run_state import RunLimits
        RunLimits(continuous=True, **value.get("limits", {}))
        return cls(**value)

    def as_dict(self):
        return {"format": FORMAT, **asdict(self), "wiki_ids": list(self.wiki_ids)}


def prepare_selection(selection, invocation, configuration=None):
    """Transform the selected workflow only; retain Agent, Task and Execution."""
    from aka.contracts.startup import Invocation
    from aka.contracts.workflow_launch import decode_launch, encode_launch
    config = SchemaConfig.load(configuration)
    row = selection.composition.row("workflow")
    if row is None or row.disabled or row.name != "aka.workflow.light.plugin":
        raise ValueError("Schema requires the native Light workflow; retired strict/dynamic selections cannot resume")
    launch = decode_launch(invocation.argv)
    request = launch.request
    explicit = launch.provided_settings
    settings = request.settings
    updates = {}
    defaults = {"framework_baseline": "never", "convert_after": 0,
                "setup_timeout": int(config.limits.get("per_call_active_seconds", 3600))}
    if config.continuous:
        defaults["max_iters"] = 0
    for key, value in defaults.items():
        if explicit is not None and key not in explicit:
            updates[key] = value
    settings = replace(settings, **updates)
    if settings.framework_baseline != "never" or settings.framework_baseline_reviewers or settings.convert_after != 0:
        raise ValueError("Schema requires framework_baseline='never', no baseline reviewers, convert_after=0")
    if not settings.framework:
        raise ValueError("Schema requires an explicit native framework")
    if type(settings.max_iters) is not int or settings.max_iters < 0:
        raise ValueError("Schema max_iters must be a nonnegative integer")
    if settings.max_iters == 0 and not config.continuous:
        raise ValueError("max_iters=0 requires explicit Schema continuous=true")
    native = dict(row.config)
    for key in ("search", "goal_episodes", "generate_memory", "wiki", "research_records"):
        if native.get(key):
            raise ValueError("Schema requires serial fixed-resource Light; incompatible option: " + key)
    native.update(search={}, goal_episodes=False, generate_memory=False, wiki=False,
                  research_records=False, inherit_state=False)
    per_call = config.limits.get("per_call_active_seconds", 3600)
    native.setdefault("episode_timeout_seconds", per_call)
    if config.initial is not None:
        source = selection.composition.row("source")
        problem = selection.composition.row("problem")
        selected_kernel = source is not None and not source.disabled and source.config.get("kind") == "kernel"
        selected_tree = problem is not None and not problem.disabled and bool(problem.config.get("source_tree"))
        if not selected_kernel and not selected_tree:
            raise ValueError("Supplied Schema requires an explicitly selected native kernel source or source_tree; benchmark reference is not SOTA")
        if selected_kernel:
            from aka.core.filesystem import read_regular_bytes
            raw = read_regular_bytes(Path(source.config["kernel_demo"]).expanduser(), limit=64 * 1024 * 1024)
            if config.initial["source_files"] != {"kernel.py": hashlib.sha256(raw).hexdigest()}:
                raise ValueError("Supplied Schema source_files do not match the explicitly selected native kernel")
    replacements = {"workflow": replace(row, name="aka.workflow.schema.plugin",
        config={"schema": config.as_dict(), "native": native, "startup": bool(native.get("startup", True))})}
    resources = selection.composition.row("resources")
    if resources is not None:
        if resources.name != "aka.resources.plugin":
            raise ValueError("Schema requires explicit native Resources selection")
        replacements["resources"] = replace(resources, config={"plugins": [], "bundles": []})
    composition = replace(selection.composition,
        rows=tuple(replacements.get(item.id, item) for item in selection.composition.rows))
    # These are the stock launch profile's dynamic Wiki manifest roots. The
    # Schema selection replaced that resource slot with fixed packaged inputs,
    # so capture must not visit a source checkout that is no longer selected.
    repo_root = selection.composition.variables.get("repo_root")
    wiki_roots = {Path(repo_root) / path for path in ("plugins/gpu-wiki", "gpu-wiki")} if repo_root else set()
    retained_resources = []
    retained_submodules = {}
    for root, paths in selection.resources:
        paths = tuple(path for path in paths if (root / path) not in wiki_roots)
        if paths:
            retained_resources.append((root, paths))
            pinned = selection.submodules.get(str(root), {})
            retained_submodules[str(root)] = {name: sha for name, sha in pinned.items()
                if any(Path(name).is_relative_to(Path(path)) for path in paths)}
    launch = replace(launch, request=replace(request, settings=settings),
        provided_settings=None if explicit is None else tuple(sorted(set(explicit) | set(updates))))
    return replace(selection, composition=composition, resources=tuple(retained_resources),
        submodules=retained_submodules), Invocation(encode_launch(launch), invocation.environment)


def run(profile, invocation, *, configuration=None, patch_files=(), variables=None):
    from aka.bootstrap.profile import resolve_profile
    from aka.bootstrap.host import run_selection
    selection = resolve_profile(profile, patch_files=patch_files, variables=variables or {})
    return run_selection(selection, invocation, schema=configuration if configuration is not None else SchemaConfig())
