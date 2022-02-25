import json
from contextlib import contextmanager
from functools import partial, reduce
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import TYPE_CHECKING, Dict, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from atqo import parallel_map
from atqo.distributed_apis import DEFAULT_MULTI_API
from atqo.lock_stores import get_lock_store

if TYPE_CHECKING:
    import dask.dataframe as dd
    from s3path import S3Path

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
        root_path: Union["S3Path", Path, str],
        max_records: int = 0,
        group_cols: Optional[Union[str, list]] = None,
        ensure_same_cols: bool = False,
        env_parents: Optional[Dict[str, Union["S3Path", Path, str]]] = None,
        mkdirs=True,
        extra_metadata: Optional[Dict[str, _T_JSON_SERIALIZABLE]] = None,
        dask_client_address: Optional[str] = None,
        lock_store_str: Optional[str] = None,
    ):
        self.extra_metadata = extra_metadata or {}
        self._env_parents = env_parents or {}
        self._is_single_file = (not max_records) and (group_cols is None)
        self._remake_dirs = mkdirs

        _root_path = _parse_path(root_path)
        self.name = _root_path.name
        self._default_env = self._get_default_env(_root_path)
        self._current_env = self._default_env
        self._mkdirs()

        self.max_records = max_records
        self.group_cols = (
            [group_cols] if isinstance(group_cols, str) else group_cols
        )
        self._path_kls = type(self._root_path)

        self._ensure_cols = ensure_same_cols
        default_str = f"file://{TemporaryDirectory().name}"  # TODO
        self._lock_store_str = lock_store_str or default_str
        self._locks = get_lock_store(lock_store_str)
        self._base_dask_address = dask_client_address

    def extend(
        self,
        df: Union[pd.DataFrame, "dd.DataFrame"],
        missdic=None,
        try_dask=True,
    ):
        if self._ensure_cols:
            df = self._reindex_cols(df)

        if not isinstance(df, pd.DataFrame):
            assert self._get_client_address(True), f"{type(df)} needs dask"
            return df.map_partitions(
                self.extend, missdic=missdic, try_dask=False, meta={}
            ).compute()

        if self.group_cols is not None:
            return self._gb_handle(df, self.extend, try_dask=try_dask)

        if self.max_records == 0:
            lock = self._locks.acquire(self.full_path)
            return (
                self.get_full_df(try_dask=try_dask)
                .pipe(_append, df)
                .pipe(self._write_df_to_path, path=self.full_path, lock=lock)
            )

        extension_lock = self._locks.acquire(
            f"{_to_full_path(self._root_path)} - ext"
        )
        missdic = missdic or {}
        if self.n_files:
            last_path = self._get_last_full_path()
            missing = self.max_records - pd.read_parquet(last_path).shape[0]
            if missing > 0:
                missdic[last_path] = missing
        self._extend_parts(df, missdic)
        extension_lock.release()

    def batch_extend(
        self,
        df_iterator,
        dist_api=DEFAULT_MULTI_API,
        batch_size=None,
        pbar=False,
        **para_kwargs,
    ):

        parallel_map(
            partial(self.extend, try_dask=False),
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
        return parallel_map(
            partial(_map_group, fun=fun),
            self._gb_paths,
            batch_size=batch_size,
            dist_api=dist_api,
            pbar=pbar,
            **para_kwargs,
        )

    def replace_records(
        self, df: Union[pd.DataFrame, "dd.DataFrame"], by_groups=False
    ):
        """replace records in files based on index"""
        if by_groups:
            if self.group_cols is None:
                raise TypeError("only works if group cols is set")
            return self._gb_handle(df, self.replace_records)

        inds = df.index
        if not isinstance(df, pd.DataFrame):
            inds = inds.compute()
            df = df.groupby(df.index).first()
        else:
            df = df.loc[~inds.duplicated(keep="first"), :]

        missdic = {}
        for full_path in self._get_full_paths():
            lock = self._locks.acquire(full_path)
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

        return self._gb_handle(df, self.replace_all)

    def replace_all(self, df: Union[pd.DataFrame, "dd.DataFrame"]):
        """purges everything and writes df instead"""
        self.purge()
        self.extend(df)

    def purge(self):
        """purges everything"""
        for p in self._get_pobjs():
            p.unlink()

    def get_full_df(self, try_dask=True):
        if try_dask and (self._get_client_address(True) is not None):
            return self.get_full_ddf().compute()
        return reduce(
            _reducer,
            self._get_full_paths(),
            pd.DataFrame(),
        )

    def get_full_ddf(self):
        import dask.dataframe as dd

        if self.n_files:
            return dd.read_parquet(self._get_full_paths())
        return dd.from_pandas(pd.DataFrame(), npartitions=1)

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
        if lock is None:
            lock = self._locks.acquire(path)
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
            _reducer(df.iloc[start_rec:end, :], fpath).pipe(
                self._write_df_to_path, path=fpath
            )
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
        return [*map(_to_full_path, self._sorted_paths)]

    def _get_last_full_path(self):
        return self._get_full_paths()[-1]

    def _next_full_path(self):
        while True:
            fname = "file-{:020d}{}".format(self.n_files + 1, EXTENSION)
            yield _to_full_path(self._root_path / fname)

    def _gb_handle(
        self, df: Union[pd.DataFrame, "dd.DataFrame"], fun, **kwargs
    ):
        _gb = df.groupby(self.group_cols)
        if not isinstance(df, pd.DataFrame):
            _gb.apply(self._gapply, fun, meta={}, **kwargs).compute()
        else:
            _gb.apply(self._gapply, fun, **kwargs)

    def _gapply(self, gdf, fun, **kwargs):
        if gdf.empty:
            # TODO: warn here
            return
        gid = (
            gdf.iloc[[0], :]
            .reset_index()
            .loc[:, self.group_cols]
            .values[0, :]
            .astype(str)
        )
        gpath = self._path_kls(self._root_path, *gid)
        getattr(TableRepo(gpath, **self._grouped_kwargs), fun.__name__)(
            gdf, **kwargs
        )

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

    def _path_grouper(self, p: Path):
        return p.relative_to(self._root_path).parts[: len(self.group_cols)]

    def _get_client_address(self, force_init: bool):
        if (self._base_dask_address is None) and force_init:
            self._base_dask_address = _get_addr()
        return self._base_dask_address

    @property
    def _root_path(self) -> Path:
        return self._current_env_parent / self.name

    @property
    def _current_env_parent(self) -> Path:
        return _parse_path(self._env_parents[self._current_env])

    @property
    def _sorted_paths(self):
        return sorted(self._get_pobjs())

    @property
    def _gb_paths(self):
        for _, g in groupby(self._sorted_paths, self._path_grouper):
            yield list(g)

    @property
    def _grouped_kwargs(self):
        return dict(
            max_records=self.max_records,
            ensure_same_cols=self._ensure_cols,
            mkdirs=self._mkdirs,
            extra_metadata=self.extra_metadata,
            dask_client_address=self._get_client_address(force_init=False),
            lock_store_str=self._lock_store_str,
        )


def _get_addr():
    try:
        import dask.dataframe  # noqa
        from distributed.client import Client, get_client

        try:
            client = get_client()
        except ValueError:
            client = Client()
        return client.scheduler.address
    except ImportError:
        return None


def _reducer(left, right):
    return _append(left, pd.read_parquet(right))


def _append(top_df, bot_df):
    return pd.concat(
        [top_df, bot_df], ignore_index=isinstance(bot_df.index, pd.RangeIndex)
    )


def _map_group(paths, fun):
    fun(pd.concat([pd.read_parquet(p) for p in paths]))


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
