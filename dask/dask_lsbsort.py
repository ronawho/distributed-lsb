# this version seems to hang and give error
# messages about large task graphs.

# it uses the strategy described in 
# https://dask.discourse.group/t/use-map-blocks-with-function-that-returns-a-tuple/84/7
# to have 'partition' return a tuple.

import argparse
import dask
from dask.distributed import Client, wait
import dask.array as da
import numpy as np
import time

radix = 8
n_buckets = 1 << radix
n_digits = 64 // radix
mask = n_buckets - 1
trace = True

def make_structured(x_chunk, block_info=None):
    if trace:
        print("make_structured ", repr(x_chunk), block_info[0])
    ret = np.zeros(x_chunk.size, dtype='u8, u8')
    start = block_info[0]['array-location'][0][0]
    if trace:
        print("start is ", start)
    for elt,rand,i in zip(ret, x_chunk, range(x_chunk.size)):
        elt[0] = rand
        elt[1] = start+i
    return ret

def bkt(x, digit):
    #print("bkt", hex(x[0]), digit)
    ret = (x[0] >> np.uint64((radix*digit))) & np.uint64(mask)
    #print("bkt ret", hex(ret))
    return ret

# now compute the data arrays for each key from each block
def partition(x_chunk, digit):
    # generate an array-of-arrays
    # inner arrays are the data for each bucket
    #if trace:
    #    print("partition ", x_chunk, digit)
    digit = np.uint64(digit)
    # count the number in each bucket
    counts = np.zeros(n_buckets, dtype='u8')
    for x in x_chunk:
        counts[bkt(x, digit)] += 1
    # allocate the subarrays
    subarrays = []
    for c in counts:
        subarrays.append(np.zeros(c, dtype='u8, u8'))
    # store the data into the subarrays
    counts.fill(0)
    for x in x_chunk:
        b = bkt(x, digit)
        subarrays[b][counts[b]] = x
        counts[b] += 1
    # return the subarrays
    #if trace:
    #    print("partition returning ", subarrays)
    return subarrays


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, help="number of records to sort",
                        default=128*1024*1024)
    parser.add_argument("--n-workers", type=int, help="number of workers", default=16)
    parser.add_argument("--chunk_size", type=int, help="chunk size", default=None)
    args = parser.parse_args()

    n_workers=args.n_workers
    client = Client(processes=True, n_workers=n_workers)

    n = args.n
    chunk_size = args.chunk_size
    if not chunk_size:
        chunk_size = n // (n_workers*8)
    chunk_size = max(1, chunk_size)

    # run a local partition just to make sure everything works,
    # before going distributed
    test_rng = np.random.default_rng()
    test_r = test_rng.integers(0, 0xffffffffffffffff, size=10, dtype='u8')
    test_x = make_structured(test_r, [{'array-location': [(0, 10)]}])
    bkt(test_x[0], 0)
    for digit in range(n_digits):
        subarrays = partition(test_x, digit) 
        test_x = np.concatenate(subarrays, axis=0)
        #print("after digit", digit)
        #for x in test_x:
        #    print(hex(x[0]))
    for i in range(test_x.size):
        if i > 0:
            assert(test_x[i-1][0] <= test_x[i][0])

    print("Generating", n, "records of input with ", n_workers,
          "workers and chunk size ", chunk_size)
    start = time.time()

    rng = da.random.default_rng()
    r = rng.integers(0, 0xffffffffffffffff, size=n, dtype='u8',
                     chunks=chunk_size)

    # create the input data, consisting of pairs of 8-byte values
    x = da.map_blocks(make_structured, r, dtype='u8, u8')
    #x = x.persist()
    x = client.persist(x)
    wait(x)
 
    stop = time.time()
    print("Generated input in ", stop-start, " seconds")

    if trace:
        print("generated input is ", x.compute())

    print("Sorting", n, "records with", n_workers,
          "workers and chunk size", chunk_size,
          "and radix ", radix, "(", n_buckets, " buckets )")
    start = time.time()

    meta = np.array([], dtype=x.dtype)

    for digit in range(n_digits):
        print("digit", digit)

        def partition_by_digit(x_chunk):
            return partition(x_chunk, digit)

        # create an array where chunks are lists; each list contains
        # the subarrays starting with that digit 
        list_arr = x.map_blocks(partition_by_digit, dtype=x.dtype, meta=meta)
        # so that the below calls do not recompute the partition
        list_arr = client.persist(list_arr)
        #list_arr.persist()

        to_concat = [ ]
        for d in range(n_buckets):
            def get_ith(x):
                return x[d]

            to_concat.append(list_arr.map_blocks(get_ith, dtype=x.dtype, meta=meta))

        x = da.concatenate(to_concat, axis=0).rechunk(chunk_size)
        x = client.persist(x)

    print("Waiting for work")
    wait(x)
    
    stop = time.time()
    print("Sorted in ", stop-start, " seconds")

    if trace:
       tmp = s.compute()
       for x in tmp:
           print(hex(x[0]), x[1])

    exit()


