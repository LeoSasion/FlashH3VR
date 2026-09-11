"""A manifest becomes active only after all referenced cache objects are committed."""
from pathlib import Path

from h3ce.cache.keys import canonical_json
from h3ce.cache.store import atomic_write


def write_manifest(path: Path, records: list[dict], store, keys):
    store.pin(str(Path(path).resolve()), keys)
    atomic_write(path, b"".join(canonical_json(record) + b"\n" for record in records))
