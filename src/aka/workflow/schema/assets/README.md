# AKA Schema materials

AKA maintains the language and role-visible specifications in this package. `docs/` is the operational document view; `ks/` retains the selected immutable source assets for provenance. `DOCUMENT_VIEWS.json` binds each adapted document to its original bytes. `SOURCES.json` records the one-time source migration.

`wiki-index.json` lists the 18 available reference directories and their complete YAML/dispatch byte sets. Selection is explicit or deterministic from operator names; no match means no Wiki reference. Wiki current/receipt files are historical provenance and are not mounted as current task approvals. Referenced external code, review logs, Git objects and hardware evidence are not bundled or presumed verified.

The kernel catalog is a static index only. It does not authorize implicit SOTA selection, source import, reference identity or GPU claims. Sources for extraction must be separately selected and snapshotted by the host.

`release.json` freezes only `language/`, `assets/`, `prompts/` and `NOTICE.md`. Workflow, state, writer and material-publication Python modules are tracked by the installed code version and its tests; changing their implementation does not invalidate frozen role materials or an existing DesignRef. Changes to language, specifications, Wiki bytes or role prompts still require `python tools/freeze_schema_release.py` and remain detectable. A previously published design continues to refer to its own retained Resources snapshot.
