# Dask Implementation of LSD Radix Sort

`dask_lsbsort.py` is a Python and [Dask](https://www.dask.org/)
implementation of LSD Radix Sort.  However, I was not able to get it to
run well (it hangs for modest problem sizes and prints error messages
about large task graphs).

Please open a PR if you know how to fix it.

# Comparison point: Dask DataFrame Sort

While it is not an LSD Radix Sort, `sort-dataframe.py` sorts similar data
using a Dask Dataframe. This can be used to demonstrate Dask's
performance at sorting.
