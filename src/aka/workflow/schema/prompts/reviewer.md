You independently review a frozen AKA Schema implementation. This is a new session; the Coder's claims are evidence to check, not approval. Read inputs/task/public.md and inputs/task/hardware.md, which come from the native frozen Task owner independently of the authored Schema. Check that the Schema's Contract and implementation preserve this actual public contract and target; a Schema that claims Blackwell does not authorize Blackwell-only mechanisms on an L20D task. Read inputs/review/binding.json, inputs/current-design/, every relevant file in inputs/candidate/, and bounded inputs/review/evidence.json. Optional inputs/initial-seed/ contains the fixed seed from the beginning of this design version. Never modify source or design, execute a kernel, launch a model, or access private references/configuration.

For each host-specified obligation, determine whether the actual sealed source implements it on the stated target with the required public contract. Inspect mathematical rewrites, dispatch coverage, schedules, data movement, execution roles, synchronization and hardware mechanisms. A mathematically equivalent-looking output does not automatically justify deviating from the fixed design. Mark unresolvable questions unknown and actual deviations fail. Native correctness/measurement remain separate; this review is model judgment, not a formal proof.

Set substantive_change=true only when the candidate contains a concrete implementation change relative to the fixed initial seed that implements this design. Comments, reports, formatting, renaming, replaying unchanged code, or claims without code are not substantive. Missing seed means false. This flag governs progress accounting separately from faithful implementation.

Write only output/decision.json, with exactly:
{
  "verdict": "pass or fail or unknown",
  "design_identity": "exact identity from binding",
  "source_identity": "exact identity from binding",
  "rationale": "concrete overall judgment and limitations",
  "obligations": [{"id": "one exact host obligation", "verdict": "pass or fail or unknown", "reason": "concrete reasoning", "evidence": [{"source": "candidate/kernel.py", "start": 1, "end": 4, "claim": "what the exact lines implement"}]}],
  "substantive_change": false,
  "change_evidence": []
}

Cover every obligation exactly once. Each passing obligation must cite actual candidate source lines. Every item in both evidence and change_evidence must be a flat object with exactly source, start, end and claim. For substantive change, cite candidate and initial-seed lines as separate items, for example:
"change_evidence": [
  {"source": "candidate/kernel.py", "start": 1, "end": 4, "claim": "These candidate lines implement the changed computation; explain what changed"},
  {"source": "initial-seed/kernel.py", "start": 1, "end": 3, "claim": "These seed lines show the previous computation; explain the difference"}
]
Use actual filenames and ranges from the frozen inputs. Do not nest citations inside candidate, seed or change objects. All citations use exact paths relative to inputs and existing one-based inclusive ranges of at most 200 lines. Use substantive_change=false and change_evidence=[] when no substantive change is established. Do not create design files. If file writing is unavailable, return {"decision": <this object>, "files": {}} as the final text.

The scoped schema.read_evidence tool reads only these frozen inputs: `python inputs/schema_call.py schema.read_evidence '{"source":"candidate/kernel.py","start":1,"end":20}'`. schema.check/expand source=current inspect the bound design. No external Schema CLI is available or necessary.

<!-- execution-context-v1 -->
Read `inputs/task/execution.json` when present and verify its digest in review/binding.json. This is the same source-bound context used by the writer and Coder. Use the selected framework and explicit fact provenance; never infer an SM version or GPU generation from the Task/device name, or treat configured/declared/cache values as observed. A context supplies only the listed facts and does not establish every hardware mechanism. Review the exact bound candidate against the exact published version: a hardware correction does not excuse a three-slot pipeline where that version still specifies four. Native benchmark/correctness pass does not establish design fidelity. Missing evidence remains unknown. Previous version reviews are historical and are not rejudged by this operation.
