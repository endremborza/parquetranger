from pathlib import Path
from typing import Optional, Union

import dask.dataframe as dd
import pandas as pd
from s3path import S3Path

EXTENSION = ".parquet"
RECORD_COUNT_FOR_EST = 500


class TableRepo:
    f"""helps with storing parquet data in a directory

    tires dividing based on group_cols, if that is None
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
        root_path: Union[S3Path, Path, str],
        max_records: int = 0,
        group_cols: Optional[Union[str, list]] = None,
    ):
        if isinstance(root_path, str):
            if root_path.startswith("s3://"):
                self._root_path = S3Path(root_path[4:])
            else:
                self._root_path = Path(root_path)
        else:
            self._root_path = root_path

        self.max_records = max_records
        self.group_cols = (
            [group_cols] if isinstance(group_cols, str) else group_cols
        )
        self._is_single_file = (not max_records) and (group_cols is None)
        self._path_kls = type(self._root_path)

        if self._is_single_file:
            tomake = self._root_path.parent
        else:
            tomake = self._root_path
        tomake.mkdir(exist_ok=True, parents=True)

    def extend(self, df: Union[pd.DataFrame, dd.DataFrame], missdic=None):

        if self.group_cols is not None:
            return self._gb_handle(df, "extend")

        if self.max_records == 0:
            if isinstance(df, dd.DataFrame):
                df = df.compute()
            self.get_full_df().append(
                df, ignore_index=isinstance(df.index, pd.RangeIndex)
            ).to_parquet(self.full_path)
        else:
            missdic = missdic or {}
            if self.n_files:
                last_path = self._get_last_full_path()
                missing = (
                    self.max_records - pd.read_parquet(last_path).shape[0]
                )
                if missing > 0:
                    missdic[last_path] = missing
            self._extend(df, missdic)

    def replace_records(
        self, df: Union[pd.DataFrame, dd.DataFrame], by_groups=False
    ):
        """replace records in files based on index"""
        if by_groups:
            if self.group_cols is None:
                raise TypeError("only works if group cols is set")
            return self._gb_handle(df, "replace_records")

        inds = df.index
        if isinstance(df, dd.DataFrame):
            inds = inds.compute()

        missdic = {}
        for full_path in self._get_full_paths():
            odf = pd.read_parquet(full_path)
            interinds = odf.index.intersection(inds)
            interlen = len(interinds)
            if interlen == 0:
                continue
            missdic[full_path] = interlen
            odf.drop(interinds).to_parquet(full_path)

        self.extend(df, missdic)

    def replace_groups(self, df: Union[pd.DataFrame, dd.DataFrame]):
        """replace files based on file name

        only viable if `group_cols` is set
        """
        if self.group_cols is None:
            raise TypeError("only works if group cols is set")

        return self._gb_handle(df, "replace_all")

    def replace_all(self, df: Union[pd.DataFrame, dd.DataFrame]):
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
        if self.n_files:
            return dd.read_parquet(self._get_full_paths())
        return dd.from_pandas(pd.DataFrame(), npartitions=1)

    @property
    def full_path(self):
        return _to_full_path(self._get_main_pobj())

    @property
    def n_files(self):
        return len(self._get_full_paths())

    def _extend(self, df, missing_rows_dic):
        start_rec = 0
        if isinstance(df, dd.DataFrame):
            df = df.compute()
            # TODO: this needs to work better

        for fpath, missing_n in missing_rows_dic.items():
            end = start_rec + missing_n
            pd.read_parquet(fpath).append(
                df.iloc[start_rec:end, :],
                ignore_index=isinstance(df.index, pd.RangeIndex),
            ).to_parquet(fpath)
            start_rec = end
        for i, fpath in zip(
            range(start_rec, df.shape[0], self.max_records),
            self._next_full_path(),
        ):
            end = i + self.max_records
            df.iloc[i:end, :].to_parquet(fpath)

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
        n = self.n_files
        while True:
            yield _to_full_path(self._root_path / f"file-{n}{EXTENSION}")
            n += 1

    def _gb_handle(self, df: Union[pd.DataFrame, dd.DataFrame], funcname):
        _gb = df.groupby(self.group_cols)
        if isinstance(df, dd.DataFrame):
            _gb.apply(self._gapply, funcname, meta=("none", object)).compute()
        else:
            _gb.apply(self._gapply, funcname)

    def _gapply(self, gdf, funcname):
        gid = (
            gdf.iloc[[0], :]
            .reset_index()[self.group_cols]
            .values[0, :]
            .astype(str)
        )
        gpath = self._path_kls(self._root_path, *gid)
        getattr(TableRepo(gpath, self.max_records), funcname)(gdf)


def _to_full_path(pobj):
    if isinstance(pobj, S3Path):
        return pobj.as_uri()
    return str(pobj)
