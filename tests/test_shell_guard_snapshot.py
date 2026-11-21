"""The CLI's public-function snapshot must retain native shell policy helpers."""
import os
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def restored_shell(tmp_path):
    guard = Path(__file__).resolve().parents[1] / "src/aka/execution/runtime_tools/session_shell_guard.sh"
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("BASH_FUNC_") and key != "BASH_ENV"}
    # Claude's Bash snapshot retains public function names but drops `_...`.
    # Restore that projection in a fresh shell with no guard startup hook.
    captured = subprocess.run(["bash", "--noprofile", "--norc", "-c", '''
source "$1"
for name in $(declare -F | awk '{print $3}'); do
    case "$name" in _*) continue ;; esac
    declare -f "$name"
done
''', "capture", str(guard)], env=environment, text=True, capture_output=True, check=True)
    snapshot = tmp_path / "public-functions.sh"
    snapshot.write_text(captured.stdout)

    def run(command, **values):
        return subprocess.run(["bash", "--noprofile", "--norc", "-c",
                               'source "$1"\n' + command, "restore", str(snapshot)],
                              env={**environment, **values}, cwd=tmp_path,
                              text=True, capture_output=True)
    return run


def test_restored_python_keeps_allowed_execution_and_gpu_import_policy(restored_shell):
    allowed = restored_shell("python3 -c 'print(42)'")
    assert allowed.returncode == 0 and allowed.stdout.strip() == "42"
    denied = restored_shell("python3 -c 'import kernel'")
    assert denied.returncode == 126
    assert "host kernel/JIT-capable GPU imports" in denied.stderr
    assert "command not found" not in denied.stderr


def test_restored_gateway_path_guard_uses_inherited_policy(restored_shell, tmp_path):
    state = tmp_path / "gateway"
    state.mkdir()
    protected = state / "sentinel"
    protected.write_text("keep")
    result = restored_shell('rm "$ATREX_PROTECTED_GATEWAY_STATE_DIR/sentinel"',
                            ATREX_PROTECTED_GATEWAY_STATE_DIR=str(state))
    assert result.returncode == 126 and protected.read_text() == "keep"
    assert "orchestrator-owned" in result.stderr
