from itertools import product
from pathlib import Path

import pandas as pd
import pytest

from parquetranger import TableRepo
from parquetranger.core import EXTENSION

df1 = pd.DataFrame(
    {
        "A": [1, 2, 3],
        "B": ["x", "y", "z"],
        "C": [1, 2, 1],
        "C2": ["a", "a", "b"],
    },
    index=["a1", "a2", "a3"],
)
df2 = pd.DataFrame(
    {
        "A": [5, 6, 7],
        "B": ["c", "d", "e"],
        "C": [2, 1, 2],
        "C2": ["b", "b", "a"],
    },
    index=["b1", "b2", "b3"],
)

df3 = pd.DataFrame(
    {
        "A": [9, 3, 1],
        "B": ["f", "g", "h"],
        "C": [2, 1, 3],
        "C2": ["a", "a", "a"],
    },
    index=["c1", "c2", "c3"],
)

df4 = pd.DataFrame(
    {
        "A": [10, 9, 17],
        "B": ["x", "da", "ex"],
        "C": [2, 1, 3],
        "C2": ["ba", "b", "a"],
    },
    index=["b4", "b2", "b3"],
)


@pytest.mark.parametrize(
    ["gb_cols"],
    [
        ("C",),
        (["C", "C2"],),
        (["C2", "C"],),
    ],
)
def test_groupby(tmp_path, gb_cols):

    troot = tmp_path / "data"
    trepo = TableRepo(troot, group_cols=gb_cols)
    base = []
    for _df in [df1, df2, df3]:
        trepo.extend(_df)
        base.append(_df)
        for gid, gdf in pd.concat(base).groupby(gb_cols):
            if not isinstance(gid, tuple):
                gid = (gid,)
            gpath = Path(troot, *map(str, gid)).with_suffix(EXTENSION)
            assert gdf.equals(pd.read_parquet(gpath))
            assert any(map(gdf.equals, trepo.dfs))
        conc = pd.concat(base)
        full_df = trepo.get_full_df()
        assert conc.reindex(full_df.index).equals(full_df)


def test_gb_maxrecs(tmp_path):
    troot = tmp_path / "data"
    trepo = TableRepo(troot, group_cols="C2", max_records=2)
    trepo.extend(df1)
    assert trepo.n_files == 2
    trepo.extend(df2)
    assert trepo.n_files == 4
    full_df = trepo.get_full_df()
    assert pd.concat([df1, df2]).reindex(full_df.index).equals(full_df)


@pytest.mark.parametrize(
    ["max_records", "n_files"],
    [
        (0, 1),
        (10, 1),
        (5, 2),
        (3, 3),
    ],
)
def test_extender_records(tmp_path, max_records, n_files):

    trepo = TableRepo(tmp_path, max_records=max_records)
    base = []
    for _df in [df1, df2, df3]:
        trepo.extend(_df)
        base.append(_df)

        conc = pd.concat(base)
        full_df = trepo.get_full_df()
        assert conc.reindex(full_df.index).equals(full_df)

    assert trepo.n_files == n_files


@pytest.mark.parametrize(
    ["max_records", "n_files"],
    [
        (0, 1),
        (10, 1),
        (2, 2),
    ],
)
def test_replace_records(tmp_path, max_records, n_files):
    trepo = TableRepo(tmp_path, max_records=max_records)
    trepo.replace_records(df2)

    full_df = trepo.get_full_df()
    assert df2.reindex(full_df.index).equals(full_df)

    trepo.replace_records(df4)

    full_df = trepo.get_full_df()
    new_df = pd.concat([df2.drop(df4.index, errors="ignore"), df4])
    assert new_df.reindex(full_df.index).equals(full_df)
    assert trepo.n_files == n_files

    dupind_df = pd.DataFrame(
        {"A": [2, 1], "B": ["1", "1"], "C": [1, 1], "C2": ["1", "1"]},
        index=["b2", "b2"],
    )
    trepo.replace_records(dupind_df)
    assert full_df.shape[0] == trepo.get_full_df().shape[0]


def test_gb_replace(tmp_path):

    _df1 = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["x", "x", "y"],
        },
        index=["x1", "x2", "y1"],
    )

    _df2 = pd.DataFrame(
        {
            "A": [10, 20],
            "B": ["x", "y"],
        },
        index=["x1", "y2"],
    )

    _mdf = pd.DataFrame(
        {"A": [10, 2, 3, 20], "B": ["x", "x", "y", "y"]},
        index=["x1", "x2", "y1", "y2"],
    )

    trepo = TableRepo(tmp_path, group_cols="B")
    trepo.replace_records(_df1)
    assert _df1.equals(trepo.get_full_df().sort_index())

    trepo.replace_records(_df2, by_groups=True)
    assert _mdf.equals(trepo.get_full_df().sort_index())

    trepo.replace_groups(_df1)
    assert _df1.equals(trepo.get_full_df().sort_index())


def test_bygroups_error(tmp_path):
    trepo = TableRepo(tmp_path / "fing")
    with pytest.raises(TypeError):
        trepo.replace_records(df1, by_groups=True)

    with pytest.raises(TypeError):
        trepo.replace_groups(df1)


def test_strin(tmp_path):
    _basetest(TableRepo(str(tmp_path / "data" / "subdir")))


@pytest.mark.parametrize(
    ["gcols", "max_records"],
    [*product([None, "C", ["C", "C2"], ["C2", "C"], "C2"], [0, 1])],
)
def test_part_paths(tmp_path, gcols, max_records):
    troot = tmp_path / "tab"
    trepo = TableRepo(troot, group_cols=gcols, max_records=max_records)
    trepo.extend(df1)
    if gcols is None:
        with pytest.raises(AttributeError):
            [*trepo.get_partition_paths("C")]
        return

    for part_col in gcols if isinstance(gcols, list) else [gcols]:
        gb_dic = {str(gid): gdf for gid, gdf in df1.groupby(part_col)}
        for gid, gpaths in trepo.get_partition_paths(part_col):
            gdf = pd.concat(map(pd.read_parquet, gpaths))
            assert gdf.equals(gb_dic[gid].reindex(gdf.index))


def test_cat_gb(tmp_path):
    trepo = TableRepo(tmp_path / "d", group_cols=["C"])
    trepo.extend(
        pd.DataFrame({"C": pd.Categorical(["A", "B", "A"], categories=list("ABC"))})
    )


def _basetest(trepo: TableRepo):
    base = []
    for _df in [df1, df2]:
        trepo.extend(_df)
        base.append(_df)
        conc = pd.concat(base)
        full_df = trepo.get_full_df()
        assert conc.reindex(full_df.index).equals(full_df)
    trepo.replace_all(df3)
    assert trepo.get_full_df().equals(df3)
    trepo.purge()
    assert trepo.get_full_df().empty
