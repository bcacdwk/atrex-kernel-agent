"""Small client for this role's AKA read/check bindings; no parser or CLI import."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.environ["AKA_AGENT_TOOL_CLIENT_PATH"])
from tools.client import call_tool

if len(sys.argv) != 3 or sys.argv[1] not in {"schema.check", "schema.expand", "schema.read_evidence"}:
    raise SystemExit("usage: schema_call.py schema.check|schema.expand|schema.read_evidence JSON")
payload = json.dumps(json.loads(sys.argv[2]), allow_nan=False).encode()
sys.stdout.buffer.write(call_tool(sys.argv[1], payload, workspace=Path.cwd()))
sys.stdout.buffer.write(b"\n")
