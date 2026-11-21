"""CPU-only native CLI suspension: no model, GPU, or existing tool endpoint."""
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from aka.agent.execution import control as control_module
from aka.agent.tools.protocol import receive, receive_json, send, send_json
from aka.core.processes import process_identity

SOURCE = Path(__file__).resolve().parents[1] / "src"
GUARD = SOURCE / "aka/execution/runtime_tools/session_shell_guard.sh"


def child():
    Path("client.pid").write_text(str(os.getpid()))
    spec = spec_from_file_location("native_sandbox", SOURCE / "aka/resources/tools/sandbox.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    code = module.main(sys.argv[2:])
    Path("native-return.json").write_text(json.dumps({"main_return": code, "pid": os.getpid()}))
    raise SystemExit(code)


def wait_for(predicate, label, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.002)
    raise TimeoutError(label)


def state(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return "gone"


def public_snapshot(guard, directory, environment):
    result = subprocess.run(["bash", "--noprofile", "--norc", "-c", '''
source "$1"
for name in $(declare -F | awk '{print $3}'); do
    case "$name" in _*) continue ;; esac
    declare -f "$name"
done
''', "capture", str(guard)], env=environment, text=True, capture_output=True, check=True)
    snapshot = directory / "public-functions.sh"
    snapshot.write_text(result.stdout)
    return snapshot


def observe(directory, *, monitor=True, gap=0, exit_code=0, guard=GUARD,
            snapshot=False, snippet=False, suspend=True, errexit=False):
    """B47's fake coordinator and original SessionControl, restricted to own PIDs."""
    directory.mkdir(parents=True, exist_ok=True)
    endpoint = directory / "s"
    own_pids, own_start, signals = [], {}, []
    originals = control_module.process_snapshot, control_module.os
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(endpoint))
    server.listen(1)
    server.settimeout(5)
    control = control_module.SessionControl(suspend=suspend)
    process = result = None
    assert sys.platform == "linux", "This fixture reads only owned Linux PIDs"
    control_module.process_snapshot = lambda: {
        p: r for p in own_pids if (r := process_identity(p)) is not None}

    def own_signal(pid, sig):
        assert pid in own_pids, "No signaling outside this fixture"
        assert process_identity(pid).start_token == own_start[pid]
        signals.append({"pid": pid, "signal": signal.Signals(sig).name})
        os.kill(pid, sig)
        if gap and sig == signal.SIGCONT and pid == own_pids[0]:
            # Test-only forced scheduler window; no delay is used by the fix.
            time.sleep(gap)

    control_module.os = SimpleNamespace(getpid=os.getpid, kill=own_signal)
    argv = [str(Path(__file__)), "--child", "--kind", "kernel-pareto-frontier"]
    if snippet:
        code = "import runpy,sys;sys.argv=" + repr(argv) + ";runpy.run_path(sys.argv[0],run_name='__main__')"
        command = shlex.join(["python3", "-c", code])
    else:
        command = shlex.join(["python3", *argv])
    command += ' > payload.json 2> client.err && printf "FOLLOWUP=ran\\n"'
    shell = ("set " + ("-m" if monitor else "+m") + "\n"
             + ("set -e\n" if errexit else "") + command + "\n"
             'status=$?\nprintf "COMMAND_STATUS=%s\\n" "$status"\n'
             'case $- in *m*) printf "MONITOR=on\\n" ;; *) printf "MONITOR=off\\n" ;; esac\n'
             'IFS= read -r release\njobs -l\n')
    environment = {"PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin", "LANG": "C",
                   "PYTHONPATH": str(SOURCE), "PYTHONDONTWRITEBYTECODE": "1",
                   "AKA_AGENT_TOOL_ENDPOINT": str(endpoint), "AKA_AGENT_TOOL_TOKEN": "fake-coordinator-only"}
    if snapshot:
        startup = public_snapshot(guard, directory, environment)
        shell = "source " + shlex.quote(str(startup)) + "\n" + shell
    else:
        environment["BASH_ENV"] = str(guard)
    try:
        process = subprocess.Popen(["bash", "--noprofile", "--norc", "-c", shell],
            cwd=directory, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True)
        own_pids.append(process.pid)
        own_start[process.pid] = process_identity(process.pid).start_token
        client_pid = int(wait_for(lambda: (directory / "client.pid").read_text()
                                 if (directory / "client.pid").exists() else None, "client pid"))
        own_pids.append(client_pid)
        own_start[client_pid] = process_identity(client_pid).start_token
        parent_pid = process_identity(client_pid).ppid
        if parent_pid != process.pid:
            # The -c guard uses a pipeline subshell. Its PID comes only from
            # this fixture's live client, never from a host process census.
            parent = process_identity(parent_pid)
            assert parent.ppid == process.pid and parent.pgid == process.pid
            own_pids.append(parent_pid)
            own_start[parent_pid] = parent.start_token
        control.clock.start()
        control.attach(process.pid)
        with server.accept()[0] as connection:
            connection.settimeout(5)
            with connection.makefile("rb", buffering=0) as reader, connection.makefile("wb", buffering=0) as writer:
                receive_json(reader, "session")
                assert receive_json(reader, "tool") == {"name": "workflow.gateway"}
                channel, payload = receive(reader)
                assert channel == "request"
                assert json.loads(payload) == {"argv": ["--kind", "kernel-pareto-frontier"]}
                with control.request():
                    if suspend:
                        wait_for(lambda: all(state(p) == "T" for p in own_pids), "owned processes stopped")
                    stopped_states = {str(p): state(p) for p in own_pids}
                    before = control.clock.active_seconds()
                    time.sleep(.02)
                    frozen_clock = control.clock.active_seconds() == before
                    assert not (directory / "native-return.json").exists()
                    # A real CPU command supplies the failure code, without mapping it.
                    failure = subprocess.run(["bash", "--noprofile", "--norc", "-c", f"exit {exit_code}"],
                                             env={"PATH": "/usr/bin:/bin"}, check=False).returncode
                    body = {"best_by_shape": {}, "shape_count": 0, "winner_kernel_ids": []}
                    response = {"status": 200, "result": {"exit_code": failure,
                        "stdout": json.dumps(body) + "\n", "stderr": "fixture command failed\n" if failure else ""}}
                send(writer, "response", json.dumps(response).encode())
                send_json(writer, "complete", {})
        native = wait_for(lambda: (directory / "native-return.json").read_text()
                         if (directory / "native-return.json").exists() else None, "native return")
        wait_for(lambda: state(client_pid) in {"gone", "Z"}, "client terminal")
        stdout, stderr = process.communicate("release\n", timeout=5)
        result = {"monitor": monitor, "suspend": suspend, "snapshot": snapshot,
                  "snippet": snippet, "errexit": errexit, "guard": str(guard),
                  "injected_root_resume_gap_ms": gap * 1000, "expected_exit_code": exit_code,
                  "bash_returncode": process.returncode, "shell_stdout": stdout, "shell_stderr": stderr,
                  "client_stderr": (directory / "client.err").read_text(),
                  "payload": json.loads((directory / "payload.json").read_text()),
                  "native_main": json.loads(native), "own_pids": list(own_pids),
                  "states_inside_request": stopped_states, "clock_frozen": frozen_clock, "signals": signals}
    finally:
        control_module.process_snapshot, control_module.os = originals
        control.close()
        for pid in reversed(own_pids):
            identity = process_identity(pid)
            if identity is None or identity.start_token != own_start.get(pid):
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process is not None:
            process.communicate(timeout=5)
        server.close()
        endpoint.unlink(missing_ok=True)
        if result is not None:
            result["cleanup_states"] = {str(p): state(p) for p in own_pids}
            (directory / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


@pytest.mark.skipif(sys.platform != "linux", reason="owns and inspects Linux test PIDs")
@pytest.mark.parametrize("monitor", [True, False])
@pytest.mark.parametrize("gap", [0, .02], ids=["natural", "injected"])
@pytest.mark.parametrize("exit_code", [0, 7, 147])
@pytest.mark.parametrize("snapshot,snippet", [(False, False), (True, False), (True, True)],
                         ids=["bash-env-script", "snapshot-script", "snapshot-snippet"])
def test_native_tool_waits_for_termination(tmp_path, monitor, gap, exit_code, snapshot, snippet):
    result = observe(tmp_path, monitor=monitor, gap=gap, exit_code=exit_code,
                     snapshot=snapshot, snippet=snippet)
    assert result["native_main"]["main_return"] == exit_code
    assert f"COMMAND_STATUS={exit_code}\n" in result["shell_stdout"]
    assert ("FOLLOWUP=ran\n" in result["shell_stdout"]) == (exit_code == 0)
    assert f"MONITOR={'on' if monitor else 'off'}\n" in result["shell_stdout"]
    assert result["bash_returncode"] == 0
    assert result["shell_stderr"] == ""
    assert result["client_stderr"] == ("fixture command failed\n" if exit_code else "")
    assert result["payload"] == {"best_by_shape": {}, "shape_count": 0, "winner_kernel_ids": []}
    assert set(result["states_inside_request"].values()) == {"T"}
    assert result["clock_frozen"]
    assert set(result["cleanup_states"].values()) == {"gone"}


if __name__ == "__main__":
    assert sys.argv[1] == "--child"
    child()
