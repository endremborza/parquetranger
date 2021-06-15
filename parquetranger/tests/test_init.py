from parquetranger import __version__, TableRepo


def test_import():
    assert isinstance(__version__, str)

def test_mkdirs(tmp_path):
    npth = tmp_path / "not-yet"
    TableRepo(npth / "sg", mkdirs=False)
    assert not npth.exists()
    TableRepo(npth / "sg")
    assert npth.exists()