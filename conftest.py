import pytest

from distributed import Client, LocalCluster

W_OPTION = "workers"

def pytest_addoption(parser):
    parser.addoption(f"--{W_OPTION}", default=3, help="number of dask/ray workers")


@pytest.fixture(scope="session")
def dask_client(pytestconfig):
    wcount = int(pytestconfig.getoption(W_OPTION))
    with LocalCluster(n_workers=wcount) as cluster, Client(cluster) as client:
        yield client
