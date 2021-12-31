from itertools import product

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from redis import Redis

from parquetranger import TableRepo
from parquetranger.parallel import dask_para


@pytest.mark.parametrize(
    ["batches", "rowcount", "max_records", "group_cols"],
    product([5, 10, 50], [10, 100, 1000], [0, 90, 900], [None, "C"]),
)
def test_para_extend(
    tmp_path, batches, rowcount, max_records, group_cols, client
):
    rng = np.random.RandomState(42069)
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

    troot = tmp_path / "data"
    trepo1 = TableRepo(
        troot,
        group_cols=group_cols,
        max_records=max_records,
        lock_store_loader=Redis,
    )
    dask_para(test_dfs, trepo1.extend, batchsize=4, client=client)
    assert_frame_equal(trepo1.get_full_df().pipe(_fit), test_df)
