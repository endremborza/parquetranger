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
