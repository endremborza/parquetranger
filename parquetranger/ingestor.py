import json
from dataclasses import dataclass, field
from hashlib import md5
from pathlib import Path
from types import NoneType
from typing import Optional
from uuid import uuid4

from atqo import acquire_lock

from .core import RecordWriter, TableRepo

ATOM_TYPES = (int, float, str, bool, NoneType)

COMP_TYPES = (list, dict)

SCHEMA_PREFIX = "schema"
KEY_PREFIX = "key"
LISTDIR = "list"
ATOM_DIR = "atoms"
ATOM_KEY = "element"

parent_id_key = "__parent_id"  # TODO: WIP


@dataclass
class ObjIngestor:
    root: Path
    size_limit = 1_000_000
    root_id_key: Optional[str] = None
    force_key: bool = False
    forward_uuids: bool = False

    writers: dict[tuple, RecordWriter] = field(default_factory=dict, init=False)
    keydic: dict[str, str] = field(default_factory=dict, init=False)

    # TODO: better memory management
    total_atoms: int = 0
    largest_size: int = 0
    # largest_key: str = ""
    # TODO: maybe some more complex key system for relations

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.dump_all()

    def ingest(self, obj, parents=(), parent_id=None):
        if isinstance(obj, list):
            for e in obj:
                self.ingest(e, (*parents, LISTDIR), parent_id)
            return
        if isinstance(obj, ATOM_TYPES):
            return self.ingest({ATOM_KEY: obj}, (*parents, ATOM_KEY), parent_id)
        if not obj:
            return
        # level = len(parents) todo for key forwarding
        comp_elems = {}
        type_map = {}
        atoms = {}
        if parent_id is not None:
            obj[parent_id_key] = parent_id
        for k, v in obj.items():
            t = type(v)
            if t in ATOM_TYPES:
                type_map[k] = t.__name__
                atoms[k] = v
            else:
                comp_elems[k] = v
        record_id = atoms.get(self.root_id_key)
        if (record_id is None) and self.force_key:  # only to root / selective to level
            record_id = uuid4().hex
            atoms[self.root_id_key] = record_id
            type_map[self.root_id_key] = type(record_id).__name__

        writer = self._get_writer(parents, type_map)
        writer.add_to_batch(atoms)
        for k, v in comp_elems.items():
            key_code = _m5(k, KEY_PREFIX)
            self.keydic[key_code] = k
            self.ingest(v, (*parents, key_code), record_id)

    def dump_largest(self):
        pass

    def dump_all(self):
        for writer in self.writers.values():
            writer.close()
        key_map_path = self.root / "key-map.json"
        map_lock = acquire_lock(key_map_path)
        try:
            if key_map_path.exists():
                self.keydic.update(json.loads(key_map_path.read_text()))
            if key_map_path.parent.exists():
                key_map_path.write_text(json.dumps(self.keydic))
        finally:
            map_lock.release()

    def _get_writer(self, parents, type_map) -> RecordWriter:
        schema_code = _m5(json.dumps(type_map, sort_keys=True), SCHEMA_PREFIX)
        key = (*parents, schema_code)
        writer = self.writers.get(key)
        if writer is None:
            reclim = self.size_limit // len(type_map)
            trepo = TableRepo(Path(self.root, *key), max_records=reclim)
            writer = RecordWriter(trepo, record_limit=self.size_limit)
            self.writers[key] = writer
        return writer


def _m5(s: str, prefix: str):
    return f"{prefix}-{md5(s.encode()).hexdigest()[:9]}"
