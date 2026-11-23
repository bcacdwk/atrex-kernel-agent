"""Run the real projected-proc helper from a fresh, cache-free source tree."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "src"


def inventory(root):
    return {
        str(path.relative_to(root)): (
            path.lstat().st_mode,
            os.readlink(path) if path.is_symlink() else
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        )
        for path in root.rglob("*")
    }


def probe(source, workspace, bwrap, case):
    from aka.agent.execution import proc_view

    assert Path(proc_view.__file__).resolve() == source / "aka/agent/execution/proc_view.py"
    environment = dict(os.environ)
    assert "PYTHONDONTWRITEBYTECODE" not in environment
    assert sys.dont_write_bytecode  # The parent flag alone must not mask this regression.
    argv = [bwrap, "--die-with-parent", "--unshare-user", "--unshare-pid",
            "--ro-bind", "/", "/", "--dev", "/dev", "--tmpfs", "/run",
            "--ro-bind", proc_view.PROC_SOURCE, "/proc", "--dir", "/run/aka-proc",
            "--ro-bind", proc_view.SHIM_SOURCE, proc_view.SHIM_TARGET,
            "--bind", str(workspace), str(workspace), "--chdir", str(workspace)]
    code = 7 if case == "failure" else 0
    shell = '''
test "$PROC_VIEW_TEST_VALUE" = fixture-value || exit 81
cat /proc/self/stat > observed.stat
cat /proc/self/status > observed.status
printf 'fixture stdout\n'
printf 'fixture stderr\n' >&2
: > ready
while [ ! -f release ]; do sleep 0.025; done
exit "$1"
'''
    launch = proc_view.prepare_launch(argv, ["/bin/sh", "-c", shell, "probe", str(code)],
        environment, workspace, provider_executable="/bin/sh")
    assert launch.environment == environment
    view = Path(launch.command[launch.command.index("--view") + 1])
    initial = inventory(source)
    assert not list(source.rglob("__pycache__"))
    process = None
    try:
        process = subprocess.Popen(launch.command, env=launch.environment, cwd=workspace,
            pass_fds=launch.pass_fds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        launch.release_arguments()
        deadline = time.monotonic() + 10
        while not (workspace / "ready").exists():
            assert process.poll() is None, "helper exited before projected proc became ready"
            assert time.monotonic() < deadline, "projected proc readiness timed out"
            time.sleep(.01)
        stat = (workspace / "observed.stat").read_text()
        status = (workspace / "observed.status").read_text().splitlines()
        assert stat.split()[0] == "1"
        assert "Pid:\t1" in status and "NSpid:\t1" in status
        assert (view / "1/cgroup").read_text() == "0::/\n"
        assert (view / "1/exe").resolve() == Path("/bin/sh").resolve()
        assert (view / "1/cmdline").read_bytes()
        assert (view / "1/task/1/children").is_file()
        assert all((view / name).is_file() for name in proc_view.PUBLIC_STATISTICS)
        children = Path(f"/proc/{process.pid}/task/{process.pid}/children")
        child_pids = [int(value) for value in children.read_text().split()]
        assert child_pids
        if case in {"sigterm", "sigint", "timeout"}:
            if case == "timeout":
                try:
                    process.communicate(timeout=.05)
                except subprocess.TimeoutExpired:
                    pass
                else:
                    raise AssertionError("helper did not remain active until cancellation")
            cancel = signal.SIGINT if case == "sigint" else signal.SIGTERM
            process.send_signal(cancel)
            code = 128 + cancel
        else:
            (workspace / "release").touch()
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == code
        assert stdout == "fixture stdout\n" and stderr == "fixture stderr\n"
        assert all(not Path(f"/proc/{pid}").exists() for pid in child_pids)
        final = inventory(source)
        print(json.dumps({"case": case, "returncode": process.returncode,
            "source_origin_matches": True, "metadata_preserved": True,
            "stdout_stderr_preserved": True, "child_reaped": True,
            "new_cache_directories": len(list(source.rglob("__pycache__"))),
            "new_pyc_files": len(list(source.rglob("*.pyc"))),
            "source_tree_unchanged": initial == final}), flush=True)
        assert initial == final, "helper wrote to its source tree"
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)
        launch.close()
        assert not view.exists()


@unittest.skipUnless(sys.platform == "linux", "projected proc requires Linux")
class ProcViewBytecodeTest(unittest.TestCase):
    def check_case(self, case):
        bwrap = os.environ.get("AKA_PROC_VIEW_TEST_BWRAP") or shutil.which("bwrap")
        if not bwrap:
            self.skipTest("bwrap is required for a real projected-proc launch")
        with tempfile.TemporaryDirectory(prefix="aka-proc-bytecode-test-") as temporary:
            root = Path(temporary)
            source = root / "src"
            for relative in ("aka/__init__.py", "aka/agent/__init__.py",
                    "aka/agent/execution/__init__.py", "aka/agent/execution/launch.py",
                    "aka/agent/execution/proc_view.py", "aka/agent/execution/proc_maps.c"):
                target = source / relative
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                target.write_bytes((SOURCE / relative).read_bytes())
                target.chmod(0o600)
            workspace = root / "workspace"
            workspace.mkdir(mode=0o700)
            environment = {"PATH": "/usr/bin:/bin", "LANG": "C", "PYTHONPATH": str(source),
                "TMPDIR": str(root), "PROC_VIEW_TEST_VALUE": "fixture-value"}
            result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                "--probe", str(source), str(workspace), str(bwrap), case],
                env=environment, cwd=workspace, capture_output=True, text=True, timeout=25)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            receipt = json.loads(result.stdout)
            self.assertTrue(receipt["source_tree_unchanged"])
            self.assertEqual(receipt["new_pyc_files"], 0)
            self.assertEqual(receipt["new_cache_directories"], 0)

    def test_success(self):
        self.check_case("success")

    def test_nonzero_exit(self):
        self.check_case("failure")

    def test_sigterm(self):
        self.check_case("sigterm")

    def test_sigint(self):
        self.check_case("sigint")

    def test_timeout_cancellation(self):
        self.check_case("timeout")


if __name__ == "__main__":
    os.umask(0o077)
    if sys.argv[1:2] == ["--probe"]:
        probe(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        unittest.main()
