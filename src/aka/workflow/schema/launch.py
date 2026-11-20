"""On-only transformation of an already selected native Light composition."""
from dataclasses import replace
from pathlib import Path


def run(profile, invocation, *, configuration, patch_files=(), variables=None):
    from aka.bootstrap.profile import resolve_profile
    from aka.bootstrap.host import run_selection
    from .provider import SchemaSelection
    selection = SchemaSelection.load(configuration)
    selection.backend_factory()  # Diagnose missing/unpinned A5 before booting services.
    launch = resolve_profile(profile, patch_files=patch_files, variables=variables or {})
    row = launch.composition.row("workflow")
    if row is None or row.disabled or row.name != "aka.workflow.light.plugin":
        raise ValueError("--schema requires the native Light workflow composition")
    path = Path(configuration).expanduser().resolve(strict=True)
    fixed = {"selection": str(path), "selection_sha256": selection.source_sha256}
    updated = replace(row, name="aka.workflow.schema.plugin", config={**fixed, "startup": True})
    replacements = {"workflow": updated}
    for entry_id, name in (("agent", "aka.workflow.schema.agent_plugin"),
                           ("problem", "aka.workflow.schema.problem_plugin")):
        original = launch.composition.row(entry_id)
        if original is None or original.disabled:
            raise ValueError(f"Schema requires the original {entry_id} composition slot")
        if entry_id == "agent" and (original.name != "aka.agent.plugin"
                or original.config.get("sandbox", "none") != "none"):
            raise ValueError("this Schema native profile requires the plain Light Agent selection; "
                             "custom/sandbox Agent profiles need their own explicit qualification")
        replacements[entry_id] = replace(original, name=name, config=fixed)
    resources = launch.composition.row("resources")
    if resources is not None:
        if resources.name != "aka.resources.plugin":
            raise ValueError("Schema requires explicit original Resources selection")
        replacements["resources"] = replace(resources, config={"plugins": [], "bundles": []})
    composition = replace(launch.composition,
                          rows=tuple(replacements.get(item.id, item) for item in launch.composition.rows))
    return run_selection(replace(launch, composition=composition), invocation)
