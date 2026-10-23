"""Stream model files without loading tensor payloads."""

import hashlib
import os
import time


def atomic_replace(source, target):
    """Windows indexers/readers can briefly hold a deny-delete file handle."""
    for attempt in range(10):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 9:
                raise
            time.sleep(0.01 * (attempt + 1))


def copy_file(source, target):
    before = source.stat()
    target.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    with source.open("rb") as reader, target.open("xb") as writer:
        while block := reader.read(1024 * 1024):
            writer.write(block)
            hasher.update(block)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Model files changed during import; retry from a stable snapshot")
    return {"sha256": hasher.hexdigest(), "file_bytes": after.st_size}
