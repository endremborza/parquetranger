import random

import pandas as pd

from parquetranger import TableRepo


def test_basic_dict_writer(tmp_path):
    trepo = TableRepo(tmp_path / "test-rec")

    with trepo.get_extending_dict_batch_writer(7) as writer:
        for i in range(30):
            writer.add_to_batch({"i": i, "thing": 30})

    assert trepo.get_full_df().shape[0] == 30

    with trepo.get_replacing_dict_batch_writer(5) as writer:
        writer.add_to_batch({"i": 10, "thing": 1})

    assert trepo.get_full_df().sort_index().iloc[:2, :].to_dict("records") == [
        {"i": 10, "thing": 1},
        {"i": 1, "thing": 30},
    ]


def test_df_batch_writer(tmp_path):
    trepo = TableRepo(tmp_path / "test-df")
    dn, rn = 30, 20
    dfs = [pd.Series(range(i, i + dn)).to_frame() for i in range(rn)]
    n = dn * rn

    with trepo.get_extending_df_batch_writer(70) as writer:
        for df in dfs:
            writer.add_to_batch(df)
        assert (n - trepo.get_full_df().shape[0]) == int((n % 70 / dn) + 1 - 1e-10) * dn

    first_full = trepo.get_full_df()
    assert pd.concat(dfs, ignore_index=True).equals(first_full)

    with trepo.get_replacing_df_batch_writer(23) as writer:
        for df in [
            pd.Series(range(9), index=list(range(50, 59))[::-1]).to_frame(),
            pd.Series(range(5, 20)).to_frame(),
        ]:
            writer.add_to_batch(df)

    new_full = trepo.get_full_df()
    assert not new_full.equals(first_full)
    assert new_full.shape == first_full.shape
    assert new_full.loc[list(range(50, 59)), 0].tolist() == list(range(8, -1, -1))


def test_fix_writer(tmp_path):
    trepo = TableRepo(tmp_path / "data")

    rng = random.Random(742)

    with trepo.get_extending_fixed_dict_batch_writer(["a", "b"], 3) as writer:
        for i in range(30):
            writer.add_to_batch(
                {k: rng.random() for k in rng.sample(["a", "b", "c", "d"], 2)}
            )

    assert trepo.get_full_df().columns.tolist() == ["a", "b"]
