"""Run a canonical Workflow request through Bootstrap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

from aka.bootstrap.host import run_profile, resume
from aka.bootstrap.continuation import read, restore
from aka.contracts.startup import Invocation
from .workflow_request import register_request_options, request_arguments

from importlib.resources import files

def profile_path(value):
    if value in (None, ""):
        value = "light"
    if value == "workflow":
        return Path(str(files("aka.workflow.legacy"))) / "profiles/workflow.json"
    if value in {"observed", "sandbox", "light", "light-sandbox"}:
        try:
            path = Path(str(files("aka.workflow.light"))) / "profiles" / (value + ".json")
        except ImportError as exc:
            raise ValueError(
                f"Launch profile {value!r} requires matching atrex-aka-light; "
                "install it or explicitly select --profile workflow for Legacy") from exc
        if not path.is_file():
            raise ValueError(f"Packaged launch profile is missing: {path}")
        return path
    return Path(value).expanduser().resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="aka workflow / aka optimize",
        description="Run a public Workflow request with an explicitly selected composition.")
    parser.add_argument("--repo-root", required=True, type=Path,
                        help="explicit public tool resource root for the selected profile")
    parser.add_argument("--profile", help="launch profile (default: light; workflow selects Legacy)")
    parser.add_argument("--patch", type=Path, action="append", default=[])
    parser.add_argument("--schema", action=argparse.BooleanOptionalAction, default=None,
                        help="enable optional Schema assistance in native Light (default: off)")
    parser.add_argument("--schema-config", type=Path,
                        help="AKA Schema launch JSON (aka.schema.launch.v1)")
    parser.add_argument("--schema-path", type=Path,
                        help="complete initial Schema directory; requires pairing fields in --schema-config")
    parser.add_argument("--schema-continuous", action=argparse.BooleanOptionalAction, default=None,
                        help="continue design/implementation without an implicit total iteration cap")
    parser.add_argument("--schema-limits", type=json.loads,
                        help="explicit Schema RunLimits JSON object; finite budgets remain effective")
    # Recognize retired flags solely to diagnose them; none activates old code.
    retired = ("iterate", "task", "out", "rounds", "runtime-profile", "reference", "objective",
               "environment", "doc", "owner-account", "owner-limits", "owner-run-id")
    for name in retired:
        parser.add_argument("--schema-" + name, dest="retired_" + name.replace("-", "_"),
                            action="store_true" if name == "iterate" else "store", help=argparse.SUPPRESS)
    register_request_options(parser)
    options = parser.parse_args(argv)
    if any(getattr(options, "retired_" + name.replace("-", "_")) for name in retired):
        parser.error("retired strict/dynamic/iterate Schema options; use --schema with aka.schema.launch.v1")
    schema_options = (options.schema_config is not None or options.schema_path is not None
                      or options.schema_continuous is not None or options.schema_limits is not None)
    if schema_options and not options.schema:
        parser.error("Schema options require explicit --schema")
    try:
        arguments = request_arguments(options, parser)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    root = options.repo_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"tool resource directory does not exist: {root}")
    payload = read(os.environ)
    if payload is not None:
        if options.patch or options.profile is not None or options.schema is not None or schema_options:
            parser.error("cannot override a reconstructed launch selection")
        if options.workflow_request is None:
            parser.error("continuation requires its recorded --workflow-request")
        selection = restore(payload)
        if selection.composition.variables.get("repo_root") != str(root):
            parser.error("a different tool resource root is requested by continuation")
        return resume(Invocation(arguments))
    try:
        profile = profile_path(options.profile)
    except ValueError as error:
        parser.error(str(error))
    if options.schema:
        try:
            from dataclasses import replace
            from aka.workflow.schema.launch import SchemaConfig, run as run_schema
            config = SchemaConfig.load(options.schema_config, initial_schema=options.schema_path)
            if options.schema_continuous is not None:
                config = replace(config, continuous=options.schema_continuous)
            if options.schema_limits is not None:
                if not isinstance(options.schema_limits, dict):
                    parser.error("--schema-limits requires a JSON object")
                config = replace(config, limits={**config.limits, **options.schema_limits})
            return run_schema(profile, Invocation(arguments), configuration=config,
                patch_files=options.patch, variables={"repo_root": str(root)})
        except (ImportError, TypeError, ValueError, OSError) as error:
            parser.error(str(error))
    return run_profile(profile, Invocation(arguments),
        patch_files=options.patch, variables={"repo_root": str(root)})


if __name__ == "__main__":
    raise SystemExit(main())
