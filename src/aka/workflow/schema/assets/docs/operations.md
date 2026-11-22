# AKA Schema operations

Schema files are data. The AKA host owns all reads, checks, snapshots and publication.

- `schema.check` checks one complete directory containing `root.yaml`, optional `<id>.yaml` nodes and `dispatch.md`. It returns stable diagnostics and never edits inputs.
- `schema.expand` returns a derived effective view with the shared root Contract and only actual same-region providers. The original files remain authoritative.
- `schema.read_evidence` reads a fixed selected source with its digest and exact line range. A locator in YAML does not make an unavailable external repository available.

Use the published specifications in this directory and `../starter.yaml`. The starter is unfinished input, not a valid design or a hardware capability claim. Mathematical and Pipe expressions are statically checked and never executed.

Initial design and optimization treat length, punctuation and wrapper style as advisory. Numeric syntax, identifiers, references and full-directory closure remain mandatory. Extraction applies its stage policy and independent source review. Explicit `skip_hardware=true` skips hardware structure and related relations; it does not establish same-GPU identity or waive source evidence.

A complete replacement contains all files. The host retains raw output and any normalization changes, validates a strict reread, then publishes a manifest only after every file is durable. A returned check result alone does not grant extraction approval, correctness, benchmark coverage or promotion.

Only the current stage owner can change its allowed fields. Contract changes, complete-design selection, provider changes and final review follow the AKA extraction stage driver. Do not invoke an external CLI, dynamically load a parser or modify kernel code during design/extract/optimize.
