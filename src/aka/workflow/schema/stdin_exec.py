"""Bind a retained role input to stdin, then exec the selected native CLI.

The original Agent owns this PID/process group, timeout and stdout/stderr. No
shell, child runtime, retry loop, budget, credential lookup or output rewrite.
"""
import hashlib
import os
from pathlib import Path
import stat
import sys


def main(argv=None):
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) < 5 or args[3] != "--":
        raise ValueError("fixed input, input hash, executable hash and command required")
    source, expected_input, expected_executable, _, *command = args
    path = Path(source)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise PermissionError("native role input must be a real absolute file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise PermissionError("native role input is not an independent regular file")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            actual = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected_input:
            raise PermissionError("native role input changed before exec")
        if "sha256:" + hashlib.sha256(Path(command[0]).read_bytes()).hexdigest() != expected_executable:
            raise PermissionError("native CLI executable changed before exec")
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.dup2(descriptor, 0)
    finally:
        os.close(descriptor)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
