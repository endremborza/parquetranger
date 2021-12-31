from parquetranger.parallel import try_client

W_OPTION = "workers"

def pytest_addoption(parser):
    parser.addoption(f"--{W_OPTION}", default=3, help="number of dask workers")


def pytest_generate_tests(metafunc):
    cliet_key = "client"
    if cliet_key in metafunc.fixturenames:
        wcount = int(metafunc.config.getoption(W_OPTION))
        metafunc.parametrize(cliet_key, [try_client(wcount)])
