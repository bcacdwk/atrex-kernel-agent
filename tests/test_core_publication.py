"""CPU publication contracts, including real no-replace and process-death IO."""
import ctypes
import errno
import os
import signal
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from aka.core import publication


PAYLOAD = b"complete immutable publication\n"


def readonly(path, payload=PAYLOAD):
    path.write_bytes(payload)
    path.chmod(0o444)
    return path


def staging(path):
    return path.with_name("." + path.name + ".aka-publish")


def assert_published(path):
    value = path.lstat()
    assert stat.S_ISREG(value.st_mode)
    assert stat.S_IMODE(value.st_mode) == 0o444
    assert value.st_nlink == 1
    assert path.read_bytes() == PAYLOAD
    assert not staging(path).exists()


@pytest.fixture(params=("rename", "link"))
def backend(request, monkeypatch):
    if request.param == "link":
        monkeypatch.setattr(publication, "_rename_noreplace", lambda *args: False)
    return request.param


def test_native_exclusive_rename_moves_inode_and_refuses_existing_entry(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    readonly(source)
    inode = source.stat().st_ino
    parent = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        if not publication._rename_noreplace(parent, source.name, target.name):
            pytest.skip("exclusive rename unavailable on this filesystem")
        assert not source.exists()
        assert target.stat().st_ino == inode
        readonly(source, b"conflicting bytes")
        with pytest.raises(FileExistsError):
            publication._rename_noreplace(parent, source.name, target.name)
        assert target.read_bytes() == PAYLOAD
        assert target.stat().st_ino == inode
        assert source.read_bytes() == b"conflicting bytes"
    finally:
        os.close(parent)


@pytest.mark.parametrize("error", [errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP])
def test_only_unsupported_native_rename_uses_link_fallback(tmp_path, monkeypatch, error):
    def unavailable(*args):
        ctypes.set_errno(error)
        return -1
    monkeypatch.setattr(publication.sys, "platform", "linux")
    monkeypatch.setattr(publication.ctypes, "CDLL", lambda *a, **k: SimpleNamespace(renameat2=unavailable))
    path = tmp_path / "object"
    publication.publish_immutable(path, PAYLOAD)
    assert_published(path)


def test_missing_libc_symbol_uses_link_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(publication.sys, "platform", "linux")
    monkeypatch.setattr(publication.ctypes, "CDLL", lambda *a, **k: SimpleNamespace())
    path = tmp_path / "object"
    publication.publish_immutable(path, PAYLOAD)
    assert_published(path)


def test_other_platform_uses_link_fallback_without_loading_linux_api(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Linux-only API loaded on another platform")
    monkeypatch.setattr(publication.sys, "platform", "freebsd")
    monkeypatch.setattr(publication.ctypes, "CDLL", forbidden)
    path = tmp_path / "object"
    publication.publish_immutable(path, PAYLOAD)
    assert_published(path)


@pytest.mark.parametrize("error", [errno.EACCES, errno.EPERM, errno.EIO, errno.EXDEV])
def test_native_rename_failure_preserves_intent_without_fallback(tmp_path, monkeypatch, error):
    def failure(*args):
        ctypes.set_errno(error)
        return -1
    def forbidden(*args, **kwargs):
        pytest.fail("unexpected fallback after a real rename error")
    monkeypatch.setattr(publication.sys, "platform", "linux")
    monkeypatch.setattr(publication.ctypes, "CDLL", lambda *a, **k: SimpleNamespace(renameat2=failure))
    monkeypatch.setattr(publication.os, "link", forbidden)
    path = tmp_path / "object"
    with pytest.raises(OSError) as raised:
        publication.publish_immutable(path, PAYLOAD)
    assert raised.value.errno == error
    assert not path.exists()
    assert staging(path).read_bytes() == PAYLOAD
    assert staging(path).stat().st_nlink == 1


def test_idempotent_publication_keeps_inode_and_rejects_new_bytes(tmp_path, backend):
    path = tmp_path / "object"
    publication.publish_immutable(path, PAYLOAD)
    inode = path.stat().st_ino
    publication.publish_immutable(path, PAYLOAD)
    assert_published(path)
    assert path.stat().st_ino == inode
    with pytest.raises(ValueError, match="collision"):
        publication.publish_immutable(path, b"different bytes")
    assert_published(path)
    assert path.stat().st_ino == inode


def install_target(path, kind):
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind in ("symlink", "hardlink"):
        alias = readonly(path.with_name("external"))
        if kind == "symlink":
            path.symlink_to(alias.name)
        else:
            os.link(alias, path)
    else:
        readonly(path, b"conflicting bytes" if kind == "different" else PAYLOAD)
        if kind == "writable":
            path.chmod(0o600)


@pytest.mark.parametrize("kind", ["different", "writable", "symlink", "directory", "fifo", "hardlink"])
def test_existing_bad_target_is_preserved_and_rejected(tmp_path, kind):
    path = tmp_path / "object"
    install_target(path, kind)
    before = publication.file_signature(path.lstat())
    with pytest.raises(ValueError):
        publication.publish_immutable(path, PAYLOAD)
    assert publication.file_signature(path.lstat()) == before
    assert not staging(path).exists()


@pytest.mark.parametrize("kind", ["same", "different", "writable", "symlink", "directory", "fifo", "hardlink"])
def test_noncooperating_creator_collision_never_replaces_target(tmp_path, monkeypatch, backend, kind):
    path = tmp_path / "object"
    rename = publication._rename_noreplace
    identities = []
    def race(parent, source, destination):
        install_target(path, kind)
        identities.append(publication.file_signature(path.lstat()))
        return rename(parent, source, destination)
    monkeypatch.setattr(publication, "_rename_noreplace", race)
    if kind == "same":
        publication.publish_immutable(path, PAYLOAD)
        assert_published(path)
    else:
        with pytest.raises(ValueError):
            publication.publish_immutable(path, PAYLOAD)
        assert staging(path).read_bytes() == PAYLOAD
    assert publication.file_signature(path.lstat()) == identities[0]


def test_real_unknown_staging_hardlink_is_not_cleaned(tmp_path):
    path = tmp_path / "object"
    intent = readonly(staging(path))
    alias = tmp_path / "unknown-alias"
    os.link(intent, alias)
    with pytest.raises(ValueError, match="unknown immutable publication alias"):
        publication.publish_immutable(path, PAYLOAD)
    assert intent.stat().st_nlink == alias.stat().st_nlink == 2
    assert not path.exists()


@pytest.mark.parametrize("change", ["mode", "bytes", "hardlink"])
def test_renamed_target_is_still_verified_before_success(tmp_path, monkeypatch, change):
    path = tmp_path / "object"
    rename = publication._rename_noreplace
    def tamper(*args):
        if not rename(*args):
            pytest.skip("exclusive rename unavailable on this filesystem")
        if change == "hardlink":
            os.link(path, tmp_path / "unknown-alias")
        else:
            path.chmod(0o600)
            if change == "bytes":
                path.write_bytes(b"changed bytes")
                path.chmod(0o444)
        return True
    monkeypatch.setattr(publication, "_rename_noreplace", tamper)
    with pytest.raises(ValueError):
        publication.publish_immutable(path, PAYLOAD)
    assert path.exists()
    assert not staging(path).exists()
    if change == "hardlink":
        assert path.stat().st_nlink == 2


def test_third_link_of_legacy_publication_is_not_cleaned(tmp_path):
    path = readonly(tmp_path / "object")
    os.link(path, staging(path))
    os.link(path, tmp_path / "unknown-alias")
    with pytest.raises(ValueError):
        publication.publish_immutable(path, PAYLOAD)
    assert path.stat().st_nlink == staging(path).stat().st_nlink == 3


@pytest.mark.parametrize("retained", ["staged", "existing", "both", "linked"])
def test_recovery_file_fsync_failure_preserves_every_retained_name(tmp_path, monkeypatch, retained):
    path = tmp_path / "object"
    if retained in ("staged", "both", "linked"):
        readonly(staging(path))
    if retained in ("existing", "both"):
        readonly(path)
    elif retained == "linked":
        os.link(staging(path), path)
    names = {name: publication.file_signature(name.stat()) for name in (path, staging(path)) if name.exists()}
    fsync = os.fsync
    def fail_data(descriptor):
        value = os.fstat(descriptor)
        if stat.S_ISREG(value.st_mode) and stat.S_IMODE(value.st_mode) == 0o444:
            raise OSError(errno.EIO, "injected data fsync failure")
        fsync(descriptor)
    with monkeypatch.context() as scoped:
        scoped.setattr(publication.os, "fsync", fail_data)
        with pytest.raises(OSError, match="injected data fsync"):
            publication.publish_immutable(path, PAYLOAD)
    assert {name: publication.file_signature(name.stat()) for name in names} == names
    publication.publish_immutable(path, PAYLOAD)
    assert_published(path)


def test_directory_fsync_failure_after_move_is_recoverable(tmp_path, monkeypatch):
    path = tmp_path / "object"
    fsync = os.fsync
    def fail_committed_directory(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) and path.exists() and not staging(path).exists():
            raise OSError(errno.EIO, "injected directory fsync failure")
        fsync(descriptor)
    with monkeypatch.context() as scoped:
        scoped.setattr(publication.os, "fsync", fail_committed_directory)
        with pytest.raises(OSError, match="injected directory fsync"):
            publication.publish_immutable(path, PAYLOAD)
    assert path.read_bytes() == PAYLOAD
    publication.publish_immutable(path, PAYLOAD)
    assert_published(path)


CRASH_SCRIPT = r'''
import os, signal, stat, sys
from pathlib import Path
from aka.core import publication as p
path, boundary = Path(sys.argv[1]), sys.argv[2]
def die():
    os.kill(os.getpid(), signal.SIGKILL)
if boundary == "partial":
    def partial(parent, name, payload, mode):
        fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=parent)
        os.write(fd, payload[:8])
        die()
    p._stage = partial
elif boundary in ("before-data-fsync", "after-data-fsync"):
    original = os.fsync
    def fsync(fd):
        value = os.fstat(fd)
        if stat.S_ISREG(value.st_mode) and stat.S_IMODE(value.st_mode) == 0o444:
            if boundary == "after-data-fsync":
                original(fd)
            die()
        original(fd)
    p.os.fsync = fsync
elif boundary in ("before-rename", "after-rename"):
    original = p._rename_noreplace
    def rename(*args):
        if boundary == "after-rename":
            if not original(*args):
                raise RuntimeError("exclusive rename unavailable")
        die()
    p._rename_noreplace = rename
elif boundary == "after-link":
    p._rename_noreplace = lambda *args: False
    original = os.link
    def link(*args, **kwargs):
        original(*args, **kwargs)
        die()
    p.os.link = link
p.publish_immutable(path, b"complete immutable publication\n")
raise AssertionError("crash boundary not reached")
'''


@pytest.mark.parametrize("boundary", ["partial", "before-data-fsync", "after-data-fsync",
                                      "before-rename", "after-rename", "after-link"])
def test_sigkill_then_same_request_recovers_without_bad_aliases(tmp_path, boundary):
    path = tmp_path / "object"
    result = subprocess.run([sys.executable, "-c", CRASH_SCRIPT, str(path), boundary],
                            capture_output=True, timeout=20)
    assert result.returncode == -signal.SIGKILL, result.stderr.decode()
    if boundary == "after-rename":
        assert path.exists() and not staging(path).exists()
    elif boundary == "after-link":
        assert path.stat().st_nlink == staging(path).stat().st_nlink == 2
    else:
        assert not path.exists() and staging(path).exists()
    publication.publish_immutable(path, PAYLOAD)
    assert_published(path)


@pytest.mark.parametrize("backend", ["rename", "link"])
def test_concurrent_processes_reuse_same_immutable_inode(tmp_path, backend):
    path = tmp_path / "object"
    script = ("from aka.core import publication as p; import sys; "
              + ("p._rename_noreplace = lambda *args: False; " if backend == "link" else "")
              + "p.publish_immutable(sys.argv[1], b'complete immutable publication\\n')")
    children = [subprocess.Popen([sys.executable, "-c", script, str(path)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(6)]
    try:
        for child in children:
            _, stderr = child.communicate(timeout=20)
            assert child.returncode == 0, stderr.decode()
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
    assert_published(path)
