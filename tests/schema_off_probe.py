"""Execute the real default native entry with forbidden optional imports/I/O.

Also usable against the recorded upstream source via PYTHONPATH, producing a
normalized CPU trace for comparison. The fixture's model/evaluator are synthetic.
"""
import importlib.abc
import json
from pathlib import Path
import sys


class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == "kernelschema" or fullname.startswith(("kernelschema.", "aka.workflow.schema")):
            raise AssertionError("Optional module imported while off: " + fullname)


def no_optional_io(event, args):
    if event in {"open", "os.listdir", "os.scandir"} and args:
        path = str(args[0])
        if "KernelSchema" in path or "/optional/" in path:
            raise AssertionError("Optional checkout accessed while off: " + path)
    if event == "socket.connect":
        raise AssertionError("Network access while running the CPU off probe")


sys.meta_path.insert(0, BlockOptional())
sys.addaudithook(no_optional_io)

from schema_cpu_fixture import make_engine


def main():
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    campaign, _ = make_engine(root)
    campaign.max_iters = 2
    campaign.token_budget = 14
    campaign.handoff_resumes = 0
    reason = campaign.run()
    state = json.loads((campaign.workspace / ".atrex_long_horizon/state.json").read_text())
    assert campaign.setup_count == campaign.finish_count == 1
    exports = []
    for attempt in state["attempts"]:
        output = campaign.workspace / f".atrex_long_horizon/episodes/e{attempt['episode']:04d}/public-candidate.json"
        candidate = json.loads(output.read_text())
        exports.append({name: candidate[name] for name in
                        ("episode", "version", "unavailable", "files", "evaluations")})
    trace = {"reason": reason, "setup": campaign.setup_count, "finish": campaign.finish_count,
        "evaluations": [{"argv": list(argv), "source": source.decode()} for argv, source in campaign.evaluations],
        "calls": [{"source": call["received"].decode(), "prompt": call["prompt"].replace(str(root), "<root>")}
                  for call in campaign.agent.calls],
        "state": {name: state[name] for name in ("episodes", "accepted", "rejected", "protocol_failures", "tokens")},
        "attempts": [{name: attempt[name] for name in ("episode", "version", "status", "accepted", "tokens")}
                     for attempt in state["attempts"]],
        "exports": exports,
        "source": campaign.candidate_workspace.read_source(campaign.workspace).files["kernel.py"].decode()}
    Path(sys.argv[2]).write_text(json.dumps(trace, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
