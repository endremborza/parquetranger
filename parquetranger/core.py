import json
import pickle
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from itertools import groupby
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from atqo import acquire_lock, get_lock, parallel_map

EXTENSION = ".parquet"
DEFAULT_ENV = "default-env"
GB_KEY = "__gb_dict"


class TableRepo:
    """helps with storing, extending and reading tabular data in parquet format

    tries dividing based on group_cols, if that is None
    tries dividing based on max_records, if max records is 0
    just writes the file to root_path.parquet

    if both group_cols and max_records is given, it will create
    directories for the groups (nested directories if multiple columns given)
    """

    def __init__(
        self,
        root_path: Union[Path, str],
        max_records: int = 0,
        group_cols: Optional[Union[str, list]] = None,
        env_parents: Optional[Dict[str, Union[Path, str]]] = None,
        mkdirs=True,
        extra_metadata: Optional[Dict[str, Any]] = None,
        drop_group_cols: bool = False,
        fixed_metadata: Optional[dict] = None,
        allow_metadata_extension: bool = False,
    ):
        self.max_records = max_records
        self.drop_group_cols = drop_group_cols  # also means 'read group cols'
        self.group_cols = [group_cols] if isinstance(group_cols, str) else group_cols
        self.extra_metadata = extra_metadata or {}
        self._env_parents = env_parents or {}
        self._is_single_file = (not max_records) and (group_cols is None)
        self._remake_dirs = mkdirs

        self._fixed_meta = fixed_metadata
        self._allow_meta_extension = allow_metadata_extension

        _rp = Path(root_path)
        self.name = _rp.name
        _default_kv = filter(lambda kv: kv[1] == _rp.parent, self._env_parents.items())
        self._default_env, e_path = [*_default_kv, (DEFAULT_ENV, _rp.parent)][0]
        self._env_parents[self._default_env] = e_path
        self._current_env = self._default_env

        self._mkdirs()

    def extend(self, df: pd.DataFrame):
        if self.group_cols is not None:
            return self._gb_handle(df, self.extend)

        resolved_table = self._resolve_metadata(df)
        if self.max_records == 0:
            lock = acquire_lock(self._df_path)
            if self._df_path.exists():
                base_table = self.read_table_from_path(
                    self._df_path, lock, release=False
                )
                # BUG: https://github.com/apache/arrow/issues/34782
                if base_table.num_rows == 0:
                    out_table = resolved_table
                else:
                    out_table = pa.concat_tables([base_table, resolved_table])
            else:
                out_table = resolved_table
            return self._write_table_to_path(out_table, self._df_path, lock)

        with get_lock(f"{self.main_path} - ext"):
            self._extend_parts(resolved_table)

    def replace_records(self, df: pd.DataFrame, by_groups=False):
        """replace records in files based on index"""
        if by_groups:
            return self._gb_handle(df, self.replace_records)

        df = df.loc[~df.index.duplicated(keep="first"), :]

        for full_path in self.paths:
            lock = acquire_lock(full_path)
            odf = self.read_df_from_path(full_path, lock, release=False)
            inter_ind = odf.index.intersection(df.index)
            if len(inter_ind) == 0:
                lock.release()
                continue
            odf.loc[inter_ind, :] = df.loc[inter_ind, :]
            self._write_df_to_path(odf, path=full_path, lock=lock)
            df = df.drop(inter_ind)

        if df.shape[0] > 0:
            self.extend(df)

    def batch_extend(self, df_iterator, **para_kwargs):
        list(parallel_map(self.extend, df_iterator, **para_kwargs))

    def map_partitions(self, fun, level=None, **para_kwargs):
        _mi = int(self.max_records > 0)
        lev_ind = slice(-len(self.group_cols) - _mi, -_mi or None)
        if level is None:

            def _idf(p: Path):
                return p.parts[lev_ind]

        else:
            _idf = self._gb_cols_from_path_meta(level)

        p_iter = map(lambda t: list(t[1]), groupby(sorted(self.paths, key=_idf), _idf))

        return parallel_map(partial(self._map_paths, fun=fun), p_iter, **para_kwargs)

    def replace_groups(self, df: pd.DataFrame):
        """replace files based on file name, only viable if `group_cols` is set"""
        return self._gb_handle(df, self.replace_all)

    def replace_all(self, df: pd.DataFrame):
        """purges everything and writes df instead"""
        self.purge()
        self.extend(df)

    def purge(self):
        """purges everything"""
        for p in self.paths:
            p.unlink()
        if self._meta_path.exists():
            self._meta_path.unlink()

    def get_full_df(self) -> pd.DataFrame:
        return self.get_full_table().to_pandas()

    def get_full_table(self) -> pa.Table:
        if plist := list(self.paths):
            return pa.concat_tables(map(self.read_table_from_path, plist))
        return pa.Table.from_pydict({})

    def get_partition_paths(
        self, partition_col: str
    ) -> Iterable[tuple[str, Iterable[Path]]]:
        _getkey = self._gb_cols_from_path_meta(partition_col)

        return groupby(sorted(self.paths, key=_getkey), _getkey)

    def set_env(self, env: str):
        self._current_env = env
        self._mkdirs()

    def set_env_to_default(self):
        self.set_env(self._default_env)

    def read_table_from_path(
        self, path, lock: Optional[Lock] = None, release=True
    ) -> pa.Table:
        assert release or (lock is not None)
        if lock is None:
            lock = acquire_lock(path)
        try:
            out: pa.Table = pq.read_table(path)
        except Exception as e:  # pragma: no cover
            lock.release()
            raise e
        if release:
            lock.release()
        if not self.drop_group_cols:
            return out
        n = out.num_rows
        for k, v in (
            self._parse_metadata(out.schema.metadata or {}).get(GB_KEY, {}).items()
        ):
            out = out.append_column(k, pa.array(np.repeat(v, n)))
        return out

    def read_df_from_path(
        self, path: Path, lock: Optional[Lock] = None, release=True
    ) -> pd.DataFrame:
        return self.read_table_from_path(path, lock, release).to_pandas()

    def get_extending_dict_batch_writer(self, max_records=1_000_000):
        return RecordWriter(self, max_records)

    def get_extending_df_batch_writer(self, max_records=1_000_000):
        return DfBatchWriter(self, max_records)

    def get_replacing_dict_batch_writer(self, max_records=1_000_000):
        return RecordWriter(self, max_records, TableRepo.replace_records)

    def get_replacing_df_batch_writer(self, max_records=1_000_000):
        return DfBatchWriter(self, max_records, TableRepo.replace_records)

    @contextmanager
    def env_ctx(self, env_name):
        _base = self._current_env
        self.set_env(env_name)
        yield
        self.set_env(_base)

    @property
    def main_path(self) -> Path:
        return self._current_env_parent / self.name

    @property
    def vc_path(self) -> Path:
        multi_file = (self.max_records > 0) or self.group_cols
        return self.main_path if multi_file else self._df_path

    @property
    def paths(self) -> Iterable[Path]:
        if self._is_single_file:
            return iter([self._df_path] if self._df_path.exists() else [])

        return self.main_path.glob("**/*" + EXTENSION)

    @property
    def n_files(self) -> int:
        return len(list(self.paths))

    @property
    def dfs(self) -> Iterable[pd.DataFrame]:
        return map(self.read_df_from_path, self.paths)

    @property
    def full_metadata(self):
        return self._parse_metadata(pq.read_schema(next(self.paths)).metadata)

    def _write_table_to_path(self, table: pa.Table, path, lock: Optional[Lock] = None):
        new_meta = (table.schema.metadata or {}) | _render_metadata(self.extra_metadata)
        if lock is None:
            lock = acquire_lock(path)
        try:
            pq.write_table(table.replace_schema_metadata(new_meta), path)
        finally:
            lock.release()

    def _write_df_to_path(self, df, path, lock: Optional[Lock] = None):
        """if lock is given, it should already be acquired"""
        self._write_table_to_path(pa.Table.from_pandas(df), path, lock)

    def _extend_parts(self, table: pa.Table):
        start_rec = 0
        if self.n_files:
            last_path = sorted(self.paths)[-1]
            with get_lock(last_path):
                missing = self.max_records - _read_size_from_path(last_path)
            if missing > 0:
                start_rec = missing
                old_table = self.read_table_from_path(last_path)
                _ctab = pa.concat_tables([old_table, table.slice(0, missing)])
                self._write_table_to_path(_ctab, last_path)

        for i in range(start_rec, table.num_rows, self.max_records):
            new_path = self.main_path / f"file-{self.n_files:020d}{EXTENSION}"
            self._write_table_to_path(table.slice(i, self.max_records), new_path)

    def _map_paths(self, paths, fun):
        return fun(pd.concat(map(self.read_df_from_path, paths)))

    def _gb_cols_from_path_meta(self, key):
        def d(path: Path):
            i = -1 - int(self.max_records > 0)
            for gc in self.group_cols[::-1]:
                gid = path.parts[i]
                yield gc, gid.replace(EXTENSION, "") if i == -1 else gid
                i -= 1

        return lambda p: dict(d(p))[key]

    def _gb_handle(self, df: pd.DataFrame, fun):
        if self.group_cols is None:
            raise TypeError("only works if group cols is set")

        ignore_index = isinstance(df.index, pd.RangeIndex)
        min_table = pa.Table.from_pandas(df.iloc[:2, :].pipe(self._de_grc))
        new_fix_meta = self._get_full_meta_dict(min_table)

        for gid, gdf in df.groupby(self.group_cols):
            self._gapply(
                gdf.reset_index(drop=True) if ignore_index else gdf,
                gid,
                fun,
                new_fix_meta,
            )

    def _gapply(self, gdf: pd.DataFrame, gid_raw, fun, meta_dic):
        if gdf.empty:
            return
        gid = gid_raw if isinstance(gid_raw, tuple) else (gid_raw,)
        gb_meta = dict(zip(self.group_cols, gid))
        gb_kwargs = dict(
            max_records=self.max_records,
            mkdirs=self._remake_dirs,
            extra_metadata=self.extra_metadata | {GB_KEY: gb_meta},
            drop_group_cols=False,
            fixed_metadata=meta_dic,
        )
        gpath = Path(self.main_path, *map(str, gid))
        _gtrepo_fun = getattr(TableRepo(gpath, **gb_kwargs), fun.__name__)
        _gtrepo_fun(gdf.pipe(self._de_grc))

    def _resolve_metadata(self, df: pd.DataFrame):
        # cast the new to old types
        # add empty ones to old ones only
        table = pa.Table.from_pandas(df)
        new_dict = _schema_to_dic(table.schema)
        full_dict = self._get_full_meta_dict(table)
        if new_dict != full_dict:
            return _cast_table(table, full_dict)
        return table

    def _get_full_meta_dict(self, new_table: pa.Table):
        metafix_lock = acquire_lock(f"{self.main_path} - meta")
        try:
            return self._inner_meta_dict(new_table)
        finally:
            metafix_lock.release()

    def _inner_meta_dict(self, new_table: pa.Table):
        new_dict = _schema_to_dic(new_table.schema)
        if self._fixed_meta is not None:
            old_dict = self._fixed_meta
        else:
            first_path = self._meta_path
            first_lock = acquire_lock(first_path)
            if first_path.exists():
                old_dict = _schema_to_dic(pq.read_schema(first_path))
            else:
                _pmeta = new_table.schema.metadata
                new_schema = pa.schema(new_dict.items(), metadata=_pmeta)
                rep_table = pa.Table.from_pylist([], schema=new_schema)
                pq.write_table(rep_table, first_path)
                old_dict = new_dict
            first_lock.release()
        full_dict = self._fixed_meta or (
            (new_dict | old_dict) if self._allow_meta_extension else old_dict
        )
        if (new_dict != full_dict) or (old_dict != full_dict):
            _w = f"mismatched schemas: \n{new_dict}\n{old_dict}\n{full_dict}"
            warnings.warn(_w, UserWarning)
            if full_dict.keys() - old_dict.keys():
                for path in self.paths:
                    lock = acquire_lock(path)
                    old_table = self.read_table_from_path(path, lock, release=False)
                    self._write_table_to_path(
                        _cast_table(old_table, full_dict), path, lock
                    )
        return full_dict

    def _mkdirs(self):
        if not self._remake_dirs:
            return
        self._current_env_parent.mkdir(exist_ok=True, parents=True)
        if not self._is_single_file:
            self.main_path.mkdir(exist_ok=True)

    def _parse_metadata(self, meta_dic: dict):
        my_keys = [*self.extra_metadata.keys(), GB_KEY]
        return {
            k.decode("utf-8"): (
                pickle.loads(v) if k.decode("utf-8") in my_keys else json.loads(v)
            )
            for k, v in meta_dic.items()
        }

    def _de_grc(self, df):
        return df.drop(self.group_cols, axis=1) if self.drop_group_cols else df

    @property
    def _meta_path(self):
        if self._is_single_file:
            return self._df_path
        return self.main_path / "empty.meta"

    @property
    def _df_path(self):
        return self.main_path.with_suffix(EXTENSION)

    @property
    def _current_env_parent(self) -> Path:
        return self._env_parents[self._current_env]


@dataclass
class RecordWriter:
    trepo: TableRepo
    record_limit: int = 1_000_000
    writer_function: Callable = TableRepo.extend
    _batch: list = field(default_factory=list, init=False)
    _record_count: int = field(default=0, init=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def add_to_batch(self, element):
        self._batch.append(element)
        self._record_count += self._rec_count_from_elem(element)
        if self._record_count >= self.record_limit:
            self._write()

    def close(self):
        if self._batch:
            self._write()

    def _write(self):
        self.writer_function(self.trepo, self._wrap_batch())
        self._batch = []
        self._record_count = 0

    def _wrap_batch(self):
        return pd.DataFrame(self._batch)

    def _rec_count_from_elem(self, elem):
        return 1


@dataclass
class DfBatchWriter(RecordWriter):
    def _wrap_batch(self):
        ig_ind = isinstance(self._batch[0].index, pd.RangeIndex)
        return pd.concat(self._batch, ignore_index=ig_ind)

    def _rec_count_from_elem(self, elem: pd.DataFrame):
        return elem.shape[0]


def _render_metadata(meta_dic):
    return {k: pickle.dumps(v) for k, v in meta_dic.items()}


def _schema_to_dic(sch):
    return dict(zip(sch.names, sch.types))


def _read_size_from_path(path):
    return pq.read_metadata(path).num_rows


def _cast_table(table: pa.Table, dic: dict[str, pa.DataType]):
    arrs = []
    for k, v in dic.items():
        try:
            arrs.append(table[k].cast(v))
        except KeyError:
            arrs.append(pa.array(np.repeat(None, table.num_rows), type=v))
    # TODO: pd schema.metadata here not added
    # difficult to create
    return pa.Table.from_arrays(arrs, schema=pa.schema(dic.items()))
