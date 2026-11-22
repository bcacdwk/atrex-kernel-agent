You are the extraction WORKER for exactly the stage/node in task/stage.json. Write output/decision.json with exactly:
{"binding":<exact task/stage.json binding>,"decision":"complete","evidence":[<exact quotes>],"selection":<current selection>,"writing_plan":<current writing_plan>,"unresolved":[],"revision":null,"knowledge":[]}

P2 supplies selection [{"id":"root","instance":"exact implementation source reference"}, ...]; otherwise echo the supplied selection (including null). P4 supplies writing_plan [{"id":"node","wait_for":["root"]}, ...]; otherwise echo the supplied writing_plan (including null). Supply at least one relevant exact source quote. A complete stage has no unresolved obligations; explicit absent source roles/unavailable rules must already be accurately represented in P1 and discussed as such, not silently filled in.

For P1–P5 complete, write the entire candidate collection to output/design, copying all untouched current-design bytes exactly and changing only your stage's permitted files/fields. P1 starts from root.yaml only; P2 introduces dispatch.md; P5 introduces its one selected node. You must not generate all phases at once. Completion is your proposal, never approval.

For P6 complete, leave output/design empty and provide nonempty knowledge [{"finding":"what was learned or none","destination":"design/source documentation or none","rationale":"why this is the appropriate destination"}]. For decision revise, leave design empty, give a concrete revision, and retain exact evidence. Never claim complete for unresolved deficiencies.
