import json
from contextlib import contextmanager
from functools import partial, reduce
from itertools import groupby
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from atqo import DEFAULT_MULTI_API, acquire_lock, get_lock, parallel_map

EXTENSION = ".parquet"
DEFAULT_ENV = "default-env"
_T_JSON_SERIALIZABLE = Union[str, int, list, dict]


class TableRepo:
    f"""helps with storing parquet data in a directory

    tries dividing based on group_cols, if that is None
    tries dividing based on max_records, if max records is 0
    just writes the file to root_path.parquet

    if both group_cols and max_records is given, it will create
    directories for the groups (nested directories if multiple columns given)

    if no `group_cols` is given, data will be broken up into
    0{EXTENSION}, 1{EXTENSION}, ...

    in case group cols is given replace group can be used
    otherwise either extend or replace records based on index
    """

    def __init__(
        self,
        root_path: Union[Path, str],
        max_records: int = 0,
        group_cols: Optional[Union[str, list]] = None,
        ensure_same_cols: bool = False,
        env_parents: Optional[Dict[str, Union[Path, str]]] = None,
        mkdirs=True,
        extra_metadata: Optional[Dict[str, _T_JSON_SERIALIZABLE]] = None,
    ):
        self.extra_metadata = extra_metadata or {}
        self._env_parents = env_parents or {}
        self._is_single_file = (not max_records) and (group_cols is None)
        self._remake_dirs = mkdirs
        self._ensure_cols = ensure_same_cols

        _root_path = Path(root_path)
        self.name = _root_path.name
        self._default_env = self._get_default_env(_root_path)
        self._current_env = self._default_env

        self.max_records = max_records
        self.group_cols = [group_cols] if isinstance(group_cols, str) else group_cols

        self._mkdirs()

    def extend(self, df: pd.DataFrame):
        if self._ensure_cols:
            df = self._reindex_cols(df)

        if self.group_cols is not None:
            return self._gb_handle(df, self.extend)

        if self.max_records == 0:
            lock = acquire_lock(self._df_path)
            return (
                self.get_full_df()
                .pipe(_append, df)
                .pipe(self._write_df_to_path, path=self._df_path, lock=lock)
            )

        with get_lock(f"{self.main_path} - ext"):
            self._extend_parts(df)

    def batch_extend(
        self,
        df_iterator,
        dist_api=DEFAULT_MULTI_API,
        batch_size=None,
        pbar=False,
        **para_kwargs,
    ):
        parallel_map(
            self.extend,
            df_iterator,
            batch_size=batch_size,
            dist_api=dist_api,
            pbar=pbar,
            **para_kwargs,
        )

    def map_partitions(
        self,
        fun,
        dist_api=DEFAULT_MULTI_API,
        batch_size=None,
        pbar=False,
        **para_kwargs,
    ):
        def _path_grouper(p: Path):
            return p.relative_to(self.main_path).parts[: len(self.group_cols)]

        return parallel_map(
            partial(_map_group, fun=fun),
            map(lambda t: list(t[1]), groupby(self._sorted_paths, _path_grouper)),
            batch_size=batch_size,
            dist_api=dist_api,
            pbar=pbar,
            **para_kwargs,
        )

    def replace_records(self, df: pd.DataFrame, by_groups=False):
        """replace records in files based on index"""
        if by_groups:
            return self._gb_handle(df, self.replace_records)

        df = df.loc[~df.index.duplicated(keep="first"), :]

        for full_path in self.paths:
            lock = acquire_lock(full_path)
            odf = pd.read_parquet(full_path)
            inter_ind = odf.index.intersection(df.index)
            if len(inter_ind) == 0:
                lock.release()
                continue
            odf.loc[inter_ind, :] = df.loc[inter_ind, :]
            self._write_df_to_path(odf, path=full_path, lock=lock)
            df = df.drop(inter_ind)

        if not df.empty:
            self.extend(df)

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

    def get_full_df(self):
        return reduce(_reducer, self.paths, pd.DataFrame())

    def get_partition_paths(self, partition_col: str):
        part_ind = self.group_cols.index(partition_col)
        path_indexer = part_ind - len(self.group_cols)
        if self.max_records:
            path_indexer -= 1

        def _getkey(path):
            return Path(path).parts[path_indexer]

        for gid, paths in groupby(sorted(self.paths, key=_getkey), _getkey):
            if path_indexer == -1:
                gid = gid.replace(EXTENSION, "")
            yield gid, list(paths)

    def set_env(self, env: str):
        self._current_env = env
        self._mkdirs()

    def set_env_to_default(self):
        self.set_env(self._default_env)

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
    def paths(self) -> Iterable[Path]:
        if self._is_single_file:
            return iter([self._df_path] if self._df_path.exists() else [])

        return self.main_path.glob("**/*" + EXTENSION)

    @property
    def n_files(self) -> int:
        return len(list(self.paths))

    @property
    def dfs(self) -> Iterable[pd.DataFrame]:
        return map(pd.read_parquet, self.paths)

    @property
    def full_metadata(self):
        for path in self.paths:
            return _parse_metadata(pq.read_schema(path).metadata)

    def _get_default_env(self, default_path: Path):
        def_parent = default_path.parent
        for env_name, parent in self._env_parents.items():
            if parent == def_parent:
                return env_name
        self._env_parents[DEFAULT_ENV] = def_parent
        return DEFAULT_ENV

    def _write_df_to_path(self, df, path, lock: Optional[Lock] = None):
        """if lock is given, it should already be acquired"""
        table = pa.Table.from_pandas(df)
        new_meta = table.schema.metadata | _render_metadata(self.extra_metadata)
        if lock is None:
            lock = acquire_lock(path)
        pq.write_table(table.replace_schema_metadata(new_meta), path)
        lock.release()

    def _extend_parts(self, df: pd.DataFrame):
        start_rec = 0
        if self.n_files:
            last_path = self._sorted_paths[-1]
            missing = self.max_records - pd.read_parquet(last_path).shape[0]
            if missing > 0:
                start_rec = missing
                ext_df = _reducer(df.iloc[:missing, :], last_path)
                self._write_df_to_path(ext_df, path=last_path)

        for i in range(start_rec, df.shape[0], self.max_records):
            new_path = self.main_path / f"file-{self.n_files:020d}{EXTENSION}"
            end = i + self.max_records
            # FIXME: isnt iloc inclusive?
            df.iloc[i:end, :].pipe(self._write_df_to_path, path=new_path)

    def _gb_handle(self, df: pd.DataFrame, fun):
        if self.group_cols is None:
            raise TypeError("only works if group cols is set")

        df.groupby(self.group_cols).apply(self._gapply, fun)

    def _gapply(self, gdf: pd.DataFrame, fun):
        if gdf.empty:
            # TODO: warn here
            return
        gid = gdf.iloc[[0], :].reset_index().loc[:, self.group_cols].values[0, :]
        gb_kwargs = dict(
            max_records=self.max_records,
            ensure_same_cols=self._ensure_cols,
            mkdirs=self._remake_dirs,
            extra_metadata=self.extra_metadata,
        )
        gpath = Path(self.main_path, *gid.astype(str))
        getattr(TableRepo(gpath, **gb_kwargs), fun.__name__)(gdf)

    def _reindex_cols(self, df):
        try:
            one_path = next(self.paths)
        except StopIteration:
            return df

        def _c_filter(c):
            return not (c.startswith("__index") or c in df.index.names)

        cols = list(filter(_c_filter, pq.read_schema(one_path).names))
        union = df.columns.union(cols)
        if union.difference(cols).shape[0]:
            for path in self.paths:
                lock = acquire_lock(path)
                reindexed_df = pd.read_parquet(path).reindex(union, axis=1)
                self._write_df_to_path(reindexed_df, path, lock)
        if union.difference(df.columns).shape[0]:
            return df.reindex(union, axis=1)
        return df

    def _mkdirs(self):
        if not self._remake_dirs:
            return
        self._current_env_parent.mkdir(exist_ok=True, parents=True)
        if not self._is_single_file:
            self.main_path.mkdir(exist_ok=True)

    @property
    def _sorted_paths(self):
        return sorted(self.paths)

    @property
    def _df_path(self):
        return self.main_path.with_suffix(EXTENSION)

    @property
    def _current_env_parent(self) -> Path:
        return self._env_parents[self._current_env]


def _reducer(left, right):
    return _append(left, pd.read_parquet(right))


def _append(top: pd.DataFrame, bot: pd.DataFrame):
    return pd.concat([top, bot], ignore_index=isinstance(bot.index, pd.RangeIndex))


def _map_group(paths, fun):
    return fun(pd.concat(map(pd.read_parquet, paths)))


def _render_metadata(meta_dic):
    return {k: json.dumps(v).encode("utf-8") for k, v in meta_dic.items()}


def _parse_metadata(meta_dic):
    return {k.decode("utf-8"): json.loads(v) for k, v in meta_dic.items()}
