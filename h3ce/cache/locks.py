"""OS-released file locks: killed workers cannot leave a permanently held lock."""
from __future__ import annotations

import os
import time
from pathlib import Path

from h3ce.errors import H3CEError


class FileLock:
    def __init__(self, path: Path, timeout: float = 30):
        self.path, self.timeout, self.stream = Path(path), timeout, None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.stream.write(b"\0")
            self.stream.flush()
        start = time.monotonic()
        while True:
            self.stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.monotonic() - start >= self.timeout:
                    self.stream.close()
                    raise H3CEError("E_CACHE_LOCK", f"Timed out waiting for {self.path.name}.")
                time.sleep(0.05)

    def __exit__(self, *_):
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_UN)
        finally:
            self.stream.close()

