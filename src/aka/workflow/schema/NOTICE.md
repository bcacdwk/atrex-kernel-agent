# AKA Schema source notice

The language implementation and selected specification, starter and Schema Wiki
materials were migrated once from KernelSchema commit
`dc6e2b3d318f0324ffdb578c3dcc6cf33a64c7e8`:
https://github.com/bcacdwk/KernelSchema/tree/dc6e2b3d318f0324ffdb578c3dcc6cf33a64c7e8

Original paths and SHA-256 values are recorded in `assets/SOURCES.json`. The
18 language source files retain their original functional boundaries; AKA owns
subsequent safe IO, checking policy, material selection and publication changes.
`assets/DOCUMENT_VIEWS.json` records source-to-view document transformations.
The synthetic language test fixture derives from
`schema_template/tools/tests/fixtures/minimal` at the same source commit.

The small stage permission, selection/writing-plan, directory and approval
projection checks in `extract_stages.py` are adapted from that commit's
`schema_extract/tools/artifacts.py` (SHA-256
`96c694d19ccc09a976307834b3bfc629fb8e013ec03d883e36b999a39d314737`).
AKA owns the serial native-role orchestration and receipt format; no upstream
Store, runner, scheduler or command-line implementation is imported.

No root LICENSE or NOTICE file was present in that source tree. This records
that missing upstream license declaration; it does not assert a license grant
or relicense those portions under AKA's license. No third-party kernel source
was migrated in this slice. Existing source/review references in Wiki data keep
their provenance meaning and are not evidence of a new task's approval.

Normal builds, checks, material selection and operation consume only AKA's
local files. The source URL and revision above are historical attribution,
not a checkout, installation, network or Git-object dependency.
