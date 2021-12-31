import json
from functools import cached_property
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Callable, Dict, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import redis_lock

if TYPE_CHECKING:
    import dask.dataframe as dd
    from s3path import S3Path

EXTENSION = ".parquet"
RECORD_COUNT_FOR_EST = 500
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
        root_path: Union["S3Path", Path, str],
        max_records: int = 0,
        group_cols: Optional[Union[str, list]] = None,
        ensure_same_cols: bool = False,
        env_parents: Optional[Dict[str, Union["S3Path", Path, str]]] = None,
        mkdirs=True,
        extra_metadata: Optional[Dict[str, _T_JSON_SERIALIZABLE]] = None,
        lock_store_loader: Optional[Callable] = None,
    ):
        self.extra_metadata = extra_metadata or {}
        self._env_parents = env_parents or {}
        self._is_single_file = (not max_records) and (group_cols is None)
        self._remake_dirs = mkdirs

        _root_path = _parse_path(root_path)

        self._current_env_parent = _root_path.parent
        self._env_parents[DEFAULT_ENV] = self._current_env_parent
        self.name = _root_path.name
        self._mkdirs()

        self.max_records = max_records
        self.group_cols = (
            [group_cols] if isinstance(group_cols, str) else group_cols
        )
        self._path_kls = type(self._root_path)

        self._ensure_cols = ensure_same_cols
        self._locks = LockStore(lock_store_loader)

    def extend(self, df: Union[pd.DataFrame, "dd.DataFrame"], missdic=None):
        if self._ensure_cols:
            df = self._reindex_cols(df)

        if self.group_cols is not None:
            return self._gb_handle(df, "extend")

        if not isinstance(df, pd.DataFrame):
            return df.map_partitions(
                self.extend, missdic=missdic, meta=("none", object)
            ).compute()

        if self.max_records == 0:
            lock = self._locks.get(self.full_path)
            return (
                self.get_full_df()
                .append(df, ignore_index=isinstance(df.index, pd.RangeIndex))
                .pipe(self._write_df_to_path, path=self.full_path, lock=lock)
            )

        extension_lock = self._locks.get(
            f"EXT - {_to_full_path(self._root_path)}"
        )
        missdic = missdic or {}
        if self.n_files:
            last_path = self._get_last_full_path()
            missing = self.max_records - pd.read_parquet(last_path).shape[0]
            if missing > 0:
                missdic[last_path] = missing
        self._extend_parts(df, missdic)
        extension_lock.release()

    def replace_records(
        self, df: Union[pd.DataFrame, "dd.DataFrame"], by_groups=False
    ):
        """replace records in files based on index"""
        if by_groups:
            if self.group_cols is None:
                raise TypeError("only works if group cols is set")
            return self._gb_handle(df, "replace_records")

        inds = df.index
        if not isinstance(df, pd.DataFrame):
            inds = inds.compute()
            df = df.groupby(df.index).first()
        else:
            df = df.loc[~inds.duplicated(keep="first"), :]

        missdic = {}
        for full_path in self._get_full_paths():
            lock = self._locks.get(full_path)
            odf = pd.read_parquet(full_path)
            interinds = odf.index.intersection(inds)
            interlen = len(interinds)
            if interlen == 0:
                lock.release()
                continue
            missdic[full_path] = interlen
            odf.drop(interinds).pipe(
                self._write_df_to_path, path=full_path, lock=lock
            )

        self.extend(df, missdic)

    def replace_groups(self, df: Union[pd.DataFrame, "dd.DataFrame"]):
        """replace files based on file name

        only viable if `group_cols` is set
        """
        if self.group_cols is None:
            raise TypeError("only works if group cols is set")

        return self._gb_handle(df, "replace_all")

    def replace_all(self, df: Union[pd.DataFrame, "dd.DataFrame"]):
        """purges everything and writes df instead"""
        self.purge()
        self.extend(df)

    def purge(self):
        """purges everything"""
        for p in self._get_pobjs():
            p.unlink()

    def get_full_df(self):
        return self.get_full_ddf().compute()

    def get_full_ddf(self):
        import dask.dataframe as dd

        if self.n_files:
            return dd.read_parquet(self._get_full_paths())
        return dd.from_pandas(pd.DataFrame(), npartitions=1)

    def set_env(self, env: str):
        self._current_env_parent = _parse_path(self._env_parents[env])
        self._mkdirs()

    def set_env_to_default(self):
        self.set_env(DEFAULT_ENV)

    @property
    def full_path(self):
        return _to_full_path(self._get_main_pobj())

    @property
    def n_files(self):
        return len(self._get_full_paths())

    @property
    def paths(self):
        return self._get_full_paths()

    @property
    def dfs(self):
        for p in self.paths:
            yield pd.read_parquet(p)

    @property
    def full_metadata(self):
        for path in self.paths:
            return _parse_metadata(pq.read_schema(path).metadata)

    def _write_df_to_path(self, df, path, lock: Optional[Lock] = None):
        """if lock is given, it should already be acquired"""
        table = pa.Table.from_pandas(df)
        if lock is None:
            lock = self._locks.get(path)
        pq.write_table(
            table.replace_schema_metadata(
                {
                    **table.schema.metadata,
                    **_render_metadata(self.extra_metadata),
                }
            ),
            path,
        )
        lock.release()

    def _extend_parts(self, df, missing_rows_dic):
        start_rec = 0
        for fpath, missing_n in missing_rows_dic.items():
            end = start_rec + missing_n
            pd.read_parquet(fpath).append(
                df.iloc[start_rec:end, :],
                ignore_index=isinstance(df.index, pd.RangeIndex),
            ).pipe(self._write_df_to_path, path=fpath)
            start_rec = end
        for i, fpath in zip(
            range(start_rec, df.shape[0], self.max_records),
            self._next_full_path(),
        ):
            end = i + self.max_records
            df.iloc[i:end, :].pipe(self._write_df_to_path, path=fpath)

    def _get_main_pobj(self):
        if self._is_single_file:
            return self._root_path.with_suffix(EXTENSION)
        return self._root_path

    def _get_pobjs(self):
        if self._is_single_file:
            pobj = self._get_main_pobj()
            return [pobj] if pobj.exists() else []

        return self._root_path.glob("**/*" + EXTENSION)

    def _get_full_paths(self):
        return sorted(map(_to_full_path, self._get_pobjs()))

    def _get_last_full_path(self):
        return self._get_full_paths()[-1]

    def _next_full_path(self):
        while True:
            fname = "file-{:020d}{}".format(self.n_files + 1, EXTENSION)
            yield _to_full_path(self._root_path / fname)

    def _gb_handle(self, df: Union[pd.DataFrame, "dd.DataFrame"], funcname):
        _gb = df.groupby(self.group_cols)
        if not isinstance(df, pd.DataFrame):
            _gb.apply(self._gapply, funcname, meta=("none", object)).compute()
        else:
            _gb.apply(self._gapply, funcname)

    def _gapply(self, gdf, funcname):
        gid = (
            gdf.iloc[[0], :]
            .reset_index()
            .loc[:, self.group_cols]
            .values[0, :]
            .astype(str)
        )
        gpath = self._path_kls(self._root_path, *gid)
        getattr(TableRepo(gpath, **self._grouped_kwargs), funcname)(gdf)

    def _reindex_cols(self, df):
        for p in self.paths:
            cols = [
                c
                for c in pq.read_schema(p).names
                if not (c.startswith("__index") or c in df.index.names)
            ]
            union = df.columns.union(cols)
            if union.difference(cols).shape[0]:
                for reinp in self.paths:
                    self._write_df_to_path(
                        pd.read_parquet(reinp).reindex(union, axis=1), reinp
                    )
            if union.difference(df.columns).shape[0]:
                return df.reindex(union, axis=1)
            break
        return df

    def _mkdirs(self):
        if not self._remake_dirs:
            return
        self._current_env_parent.mkdir(exist_ok=True, parents=True)
        if not self._is_single_file:
            self._root_path.mkdir(exist_ok=True)

    @property
    def _root_path(self) -> Path:
        return self._current_env_parent / self.name

    @property
    def _grouped_kwargs(self):
        return dict(
            max_records=self.max_records,
            ensure_same_cols=self._ensure_cols,
            mkdirs=self._mkdirs,
            extra_metadata=self.extra_metadata,
            lock_store_loader=self._locks.conn_loader,
        )


class LockStore:
    def __init__(self, conn_loader) -> None:
        self.conn_loader = conn_loader
        self._locks = {}

    def get(self, key) -> Lock:
        if self.conn_loader is None:
            try:
                ret = self._locks[key]
            except KeyError:
                ret = self._locks[key] = Lock()
        else:
            ret = redis_lock.Lock(self.conn, key)
        ret.acquire()
        return ret

    @cached_property
    def conn(self):
        return self.conn_loader()


def _parse_path(path):
    if isinstance(path, str):
        if path.startswith("s3://"):  # pragma: nocover
            from s3path import S3Path

            return S3Path(path[4:])
        else:
            return Path(path)
    return path


def _to_full_path(pobj):
    try:
        return pobj.as_posix()
    except AttributeError:
        return pobj.as_uri()  # pragma: nocover


def _render_metadata(meta_dic):
    return {k: json.dumps(v).encode("utf-8") for k, v in meta_dic.items()}


def _parse_metadata(meta_dic):
    return {k.decode("utf-8"): json.loads(v) for k, v in meta_dic.items()}
