from functools import partial
from itertools import product

import numpy as np
import pandas as pd
import pytest
from atqo import DEFAULT_DIST_API_KEY
from pandas.testing import assert_frame_equal

from parquetranger import TableRepo


@pytest.mark.parametrize(
    ["seed", "batches", "rowcount", "max_records", "group_cols"],
    product([742], [4, 11], [5, 20], [0, 19], [None, "C"]),
)
def test_para_extend(tmp_path, seed, batches, rowcount, max_records, group_cols):
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

    trepo1 = TableRepo(tmp_path / "dat", group_cols=group_cols, max_records=max_records)
    trepo1.batch_extend(test_dfs)
    assert_frame_equal(trepo1.get_full_df().pipe(_fit), test_df)


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


def test_sync_map(tmp_path):
    trepo = TableRepo(tmp_path / "d", group_cols=["A"])
    trepo.extend(pd.DataFrame({"A": [1, 2, 1]}))
    shapes = trepo.map_partitions(len, dist_api=DEFAULT_DIST_API_KEY)
    assert set(shapes) == {1, 2}


def _gbmapper(gdf, trepo, gcols):  # pragma: no cover
    gdf.groupby(gcols)[["A", "B"]].mean().reset_index().pipe(trepo.extend)
