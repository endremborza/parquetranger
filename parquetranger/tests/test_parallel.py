from functools import partial
from itertools import product

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from parquetranger import TableRepo


@pytest.mark.parametrize(
    ["seed", "batches", "rowcount", "max_records", "group_cols"],
    product([2, 3], [5, 10, 50], [10, 100, 1000], [0, 90, 900], [None, "C"]),
)
def test_para_extend(
    tmp_path, seed, batches, rowcount, max_records, group_cols, dask_client
):
    rng = np.random.RandomState(seed)
    test_dfs = [
        pd.DataFrame(
            {
                "A": rng.rand(rowcount),
                "B": rng.rand(rowcount),
                "C": rng.randint(100, 105, size=rowcount),
            }
        )
        for _ in range(batches)
    ]

    def _fit(_df):
        return _df.sort_values(["A", "B"]).reset_index(drop=True)

    test_df = pd.concat(test_dfs).pipe(_fit)

    trepo1 = TableRepo(
        tmp_path / "data-pmap",
        group_cols=group_cols,
        max_records=max_records,
        dask_client_address=dask_client.scheduler.address,
    )
    trepo1.batch_extend(test_dfs)
    assert_frame_equal(trepo1.get_full_df().pipe(_fit), test_df)

    for nparts in [3, 10, 30]:
        ddf = dd.from_pandas(test_df, npartitions=nparts)
        trepo2 = TableRepo(
            tmp_path / f"data-{nparts}-daskparts",
            group_cols=group_cols,
            max_records=max_records,
            dask_client_address=dask_client.scheduler.address,
        )
        trepo2.extend(ddf)
        assert_frame_equal(test_df, trepo2.get_full_df().pipe(_fit))


def test_map_partitions_extend(tmp_path):

    seed = 100
    rowcount = 10_000
    npartitions = 10
    group_cols = ["C"]
    max_records = 1000

    rng = np.random.RandomState(seed)
    troot = tmp_path / "data"
    trepo1 = TableRepo(
        troot,
        group_cols=group_cols,
        max_records=max_records,
    )

    df = pd.DataFrame(
        {
            "A": rng.rand(rowcount),
            "B": rng.rand(rowcount),
            "C": rng.randint(100, 105, size=rowcount),
        },
        index=pd.Series(range(0, rowcount)).astype(str).str.zfill(10),
    )

    ddf = dd.from_pandas(df, npartitions=npartitions)
    trepo1.extend(ddf)
    assert_frame_equal(df, trepo1.get_full_df().sort_index())


@pytest.mark.parametrize(
    ["rowcount", "max_records", "group_cols"],
    product([10, 100, 1000], [0, 90, 900], ["C", ["C", "G"]]),
)
def test_native_map_partitions(tmp_path, rowcount, max_records, group_cols):

    seed = 100

    rng = np.random.RandomState(seed)
    trepo1 = TableRepo(
        tmp_path / "d1",
        group_cols=group_cols,
        max_records=max_records,
    )

    trepo2 = TableRepo(
        tmp_path / "d2",
    )

    df = pd.DataFrame(
        {
            "A": rng.rand(rowcount),
            "B": rng.rand(rowcount),
            "C": rng.randint(100, 105, size=rowcount).astype(float),
            "G": rng.choice(["x", "y", "z"], size=rowcount),
        },
        index=pd.Series(range(0, rowcount)).astype(str).str.zfill(10),
    )
    trepo1.extend(df)
    trepo1.map_partitions(partial(_gbmapper, trepo=trepo2, gcols=group_cols))
    assert_frame_equal(
        df.groupby(group_cols)[["A", "B"]].mean().reset_index(),
        trepo2.get_full_df().sort_values(group_cols).reset_index(drop=True),
    )


def _gbmapper(gdf, trepo, gcols):
    gdf.groupby(gcols)[["A", "B"]].mean().reset_index().pipe(
        trepo.extend, try_dask=False
    )
