from itertools import product

import pandas as pd
import pytest

from parquetranger import TableRepo

inds = [
    None,
    pd.Series([1, 2], name="fing"),
    pd.MultiIndex.from_product([["A", "C"], [1]], names=["ix", "iy"]),
]


@pytest.mark.parametrize(["indices", "grow"], list(product(inds, [True])))
def test_diff_cols(tmp_path, indices, grow):
    _df1 = pd.DataFrame({"A": [1, 2], "C": ["g1", "g1"]}, index=indices)
    _df2 = pd.DataFrame({"B": [1, 2], "C": ["g2", "g2"]}, index=indices)

    trepo = TableRepo(
        tmp_path / "diffcols", group_cols="C", allow_metadata_extension=grow
    )
    trepo.extend(_df1)
    trepo.extend(_df2)

    df = trepo.get_full_df()
    if grow:
        assert _df1.columns.union(_df2.columns).isin(df.columns).all()
    else:  # TODO
        assert (df.columns == _df1.columns).all()


@pytest.mark.parametrize(["grow"], [(True,), (False,)])
def test_diff_schema(tmp_path, grow):
    _df1 = pd.DataFrame({"A": [1.2, 2.2], "C": ["g1", "g1"]})
    _df2 = pd.DataFrame({"A": [1, 2], "C": ["g2", "g2"]})

    trepo = TableRepo(
        tmp_path / "diffcols", group_cols="C", allow_metadata_extension=grow
    )
    trepo.extend(_df1)
    trepo.extend(_df2)

    df = trepo.get_full_df()
    assert _df1.columns.union(_df2.columns).isin(df.columns).all()


def test_nulls(tmp_path):
    _df1 = pd.DataFrame({"A": [1.2, 2.2], "C": [None, None]})
    _df2 = pd.DataFrame({"A": [1, 2], "C": ["g2", "g2"]})

    trepo = TableRepo(tmp_path / "diffcols")
    trepo.extend(_df1)
    trepo.extend(_df2)

    df = trepo.get_full_df()
    print(df)
    assert _df1.columns.union(_df2.columns).isin(df.columns).all()
