import pandas as pd
import pytest

from parquetranger import TableRepo


@pytest.mark.parametrize(
    ["indices"],
    [
        (None,),
        (pd.Series([1, 2], name="fing"),),
        (pd.MultiIndex.from_product([["A", "C"], [1]], names=["ix", "iy"]),),
    ],
)
def test_diff_cols(tmp_path, indices):

    _df1 = pd.DataFrame({"A": [1, 2], "C": ["g1", "g1"]}, index=indices)
    _df2 = pd.DataFrame({"B": [1, 2], "C": ["g2", "g2"]}, index=indices)

    trepo = TableRepo(tmp_path / "diffcols", group_cols="C", ensure_same_cols=True)
    trepo.extend(_df1)
    trepo.extend(_df2)

    df = trepo.get_full_df()
    assert _df1.columns.union(_df2.columns).isin(df.columns).all()


def test_diff_schema(tmp_path):

    _df1 = pd.DataFrame({"A": [1, 2], "C": ["g1", "g1"]})
    _df2 = pd.DataFrame({"A": [1.2, 2.2], "C": ["g2", "g2"]})

    trepo = TableRepo(tmp_path / "diffcols", group_cols="C", ensure_same_cols=True)
    trepo.extend(_df1)
    trepo.extend(_df2)

    df = trepo.get_full_df()
    assert _df1.columns.union(_df2.columns).isin(df.columns).all()
