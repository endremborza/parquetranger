import pandas as pd

from parquetranger import TableRepo


def test_simple_metadata(tmp_path):

    df = pd.DataFrame(
        {"d": pd.date_range("2020-01-01", "2020-01-10"), "x": range(2, 12)}
    )

    tpath = tmp_path / "data"

    meta = {"some": "thing", "other": {"thing": ["as", "dict", 10]}}

    trepo = TableRepo(tpath, extra_metadata=meta)
    trepo.replace_all(df)

    full_meta = trepo.full_metadata
    full_meta.pop("pandas")
    assert full_meta == meta
