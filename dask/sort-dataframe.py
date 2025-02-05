# this version uses Dask DataFrame to sort

import argparse
import time
import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
from dask.distributed import Client, wait

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, help="number of records to sort",
                        default=128*1024*1024)
    parser.add_argument("--n-workers", type=int, help="number of workers", default=16)
    parser.add_argument("--chunk-size", type=int, help="chunk size", default=None)
    args = parser.parse_args()

    n_workers=args.n_workers
    client = Client(processes=True, n_workers=n_workers)

    n = args.n
    chunk_size = args.chunk_size
    if not chunk_size:
        chunk_size = n // (n_workers*8)
    chunk_size = max(1, chunk_size)

    print("Generating", n, "records of input with ", n_workers,
          "workers and chunk size ", chunk_size)

    start = time.time()

    rng = da.random.default_rng()
    random_da = rng.integers(0, 0xffffffffffffffff, size=n, dtype='u8',
                             chunks=chunk_size)
    random_idx = da.arange(n, dtype='u8', chunks=chunk_size)

    by_index = da.stack([random_da, random_idx], axis=1)

    random_df = dd.from_dask_array(by_index, columns=['val', 'idx'])
    random_df = client.persist(random_df)
    wait(random_df)

    print("random_df2", "rows", len(random_df), random_df.head(10))
 
    stop = time.time()
    print("Generated input in ", stop-start, " seconds")

    print("Sorting", n, "records with", n_workers,
          "workers and chunk size", chunk_size)
    start = time.time()

    sorted_df = random_df.sort_values(by='val')
    sorted_df = client.persist(sorted_df)
    wait(sorted_df)

    stop = time.time()
    print("Sorted in ", stop-start, " seconds")

    print("sorted_df", "rows", len(sorted_df), sorted_df.head(10))
