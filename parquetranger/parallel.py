from itertools import islice
from multiprocessing import cpu_count

from dask.distributed import Client, LocalCluster, as_completed
from structlog import get_logger
from tqdm import tqdm

logger = get_logger()


def try_client(ncores):
    try:
        return Client.current()
    except ValueError:
        cluster = LocalCluster(n_workers=ncores)
        return Client(cluster)


def dask_para(
    params,
    fun,
    batchsize=None,
    raise_errs=True,
    pbar=True,
    static_kwargs={},
    client=None,
):
    iterator = params if hasattr(params, "__next__") else iter(params)
    ntask = batchsize or cpu_count()
    client = client or try_client(ntask)
    seq = as_completed(
        [
            client.submit(fun, e, **static_kwargs)
            for e in islice(iterator, ntask)
        ]
    )
    out = []
    wrapper = tqdm if pbar else lambda x: x
    for future in wrapper(seq):
        exc = future.exception()
        if exc is None:
            out.append(future.result())
        else:
            if raise_errs:
                future.result()
            logger.warning(exc)
        try:
            new_future = client.submit(fun, next(iterator))
        except StopIteration:
            continue
        seq.add(new_future)
    return out
