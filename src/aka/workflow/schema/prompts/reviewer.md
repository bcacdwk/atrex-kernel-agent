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

Cover every obligation exactly once. Each passing obligation must cite actual candidate source lines. For substantive change, change_evidence must cite both candidate and initial-seed source lines and explain the changed computation. All citations use exact paths relative to inputs and existing one-based inclusive ranges of at most 200 lines. Do not create design files. If file writing is unavailable, return {"decision": <this object>, "files": {}} as the final text.

The scoped schema.read_evidence tool reads only these frozen inputs: `python inputs/schema_call.py schema.read_evidence '{"source":"candidate/kernel.py","start":1,"end":20}'`. schema.check/expand source=current inspect the bound design. No external Schema CLI is available or necessary.
