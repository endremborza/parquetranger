from parquetranger import TableRepo

from .test_core import df1, df2


def test_envs(tmp_path):

    defp = tmp_path / "def"
    testp = tmp_path / "test"

    trepo = TableRepo(defp / "data", env_parents={"test": testp})

    trepo.extend(df1)
    trepo.set_env("test")
    trepo.extend(df2)

    assert trepo.get_full_df().equals(df2)
    trepo.set_env_to_default()
    assert trepo.get_full_df().equals(df1)


def test_env_ctx(tmp_path):

    fp1 = tmp_path / "fp1"
    fp2 = tmp_path / "fp2"

    trepo = TableRepo(fp1 / "data", env_parents={"e1": fp1, "e2": fp2})

    assert trepo._current_env == "e1"

    def _as1():
        assert trepo.get_full_df().equals(df1)

    def _as2():
        assert trepo.get_full_df().equals(df2)

    trepo.extend(df1)
    with trepo.env_ctx("e2"):
        trepo.extend(df2)

    _as1()
    trepo.set_env("e2")
    _as2()
    with trepo.env_ctx("e1"):
        _as1()
    _as2()

    trepo.set_env_to_default()
    _as1()
