"""Opt-in CLI adapter to KernelSchema's one finite writing/native loop."""
from pathlib import Path


def run(profile, invocation, *, task, out, schema=None, rounds=2, runtime_profile=None,
        references=(), objective=None, environment=None, docs=(), patch_files=(),
        variables=None, owner_account=None, owner_limits=None, owner_run_id=None):
    # Importing the ordinary bootstrap or fixed-Schema path never imports KS.
    from kernelschema import api
    from aka.contracts.workflow_launch import decode_launch

    launch = decode_launch(invocation.argv)
    result = api.iterate(task=task, out=out, schema=schema, rounds=rounds,
        runtime_profile=runtime_profile, references=references, objective=objective,
        environment=environment, problem=launch.request.problem.directory,
        repo_root=variables["repo_root"], platform=launch.request.settings.platform,
        profile=profile, patches=patch_files, docs=docs, _launch=launch,
        owner_account=owner_account, owner_limits=owner_limits, owner_run_id=owner_run_id)
    print(f"Schema iteration: {result['status']} ({result['stop_reason']})")
    print("Stages: " + str(Path(out).expanduser().absolute() / "iterations.json"))
    return 0 if result["status"] == "completed" else 1
