import pandas as pd

from parquetranger import TableRepo


def test_diff_cols(tmp_path):

    _df1 = pd.DataFrame({"A": [1, 2], "C": ["g1", "g1"]})
    _df2 = pd.DataFrame({"B": [1, 2], "C": ["g2", "g2"]})

    trepo = TableRepo(
        tmp_path / "diffcols", group_cols="C", ensure_same_cols=True
    )
    trepo.extend(_df1)
    trepo.extend(_df2)

    df = trepo.get_full_df()
    assert (
        df.columns.difference(_df1.columns.union(_df2.columns)).shape[0] == 0
    )


def test_diff_schema(tmp_path):

    _df1 = pd.DataFrame({"A": [1, 2], "C": ["g1", "g1"]})
    _df2 = pd.DataFrame({"A": [1.2, 2.2], "C": ["g2", "g2"]})

    trepo = TableRepo(
        tmp_path / "diffcols", group_cols="C", ensure_same_cols=True
    )
    trepo.extend(_df1)
    trepo.extend(_df2)

    df = trepo.get_full_df()
    assert (
        df.columns.difference(_df1.columns.union(_df2.columns)).shape[0] == 0
    )
