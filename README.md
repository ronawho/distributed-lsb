# Comparing Programming Tools with Distributed Radix Sorting

This repository contains code for doing a distributed parallel
least-significant-digit-first (LSD) radix sort with several
distributed-memory parallel programming frameworks. These implementations
support comparisons of these different frameworks and their productivity
and performance.

## What is LSD Radix Sort?

[Least Significant Digit Radix Sort] [1] (or LSD Radix Sort) is a
linear-time sort algorithm. It is theoretically efficient and relatively
simple to implement. Because of its simplicity, it is practical to
explore many implementations of it.

LSD Radix Sort operates in passes. Each pass stably sorts by one digit of
the data to be sorted. The first pass sorts the data by the least
significant digit (in the number 1234, 4 is the least significant digit).
The next pass sorts by the next digit (in our example 1234, 3 is the next
digit).  It proceeds in this way until all the digits have been sorted.
Each of these passes has to be stable; that is, it does not change the
order of equal elements (in this case, that is elements with the same
current digit).

## What is a Distributed-Memory Parallel Programming Framework?

A distributed-memory parallel programming framework is a programming
system that supports programs that run on a supercomputer [2] or cluster
computer [3]. It is called "distributed memory" because the supercomputer
or cluster is made up of individual nodes and each node has its own
memory.

Generally speaking, efforts to write programs for these large systems
come with challenges that a programmer needs to address in order to
achieve good performance. In particular, programs that run in parallel on
a PC or server won't necessarily perform well on a supercomputer or
cluster because the distributed-memory environment has additional
challenges. The main challenges are communication overhead and massive
parallelism.

Communication overhead is an issue because accessing data on a remote
node or coordinating with a remote node is orders of magnitude slower
than accessing local memory.
 * Local memory latency is unually measured in nanoseconds, e.g. 20ns
 * Network latency is usually measured in microseconds, e.g. 2 μs

Massive parallelism is an issue because, to use the massive amount of
parallelism available in a cluster or supercomputer, the application must
expose that much parallelism and it needs to be able to run it without
too much load imbalance. If you want to run in parallel on a PC, you
might only need to handle keeping 8 cores busy. If you want to run on a
big part of a Top500 system [4], you'll be using thousands or millions of
cores.

## Which Frameworks?

There is a directory in this repository for each framework.

### Chapel

[Chapel](https://chapel-lang.org/) is a programming language designed for
parallel computing, including distributed-memory parallel computing.

### Dask

[Dask](https://www.dask.org/) is a Python framework for parallel
computing.

### MPI

MPI ([MPI Forum][5] [OpenMPI Implementation][6] [MPICH Implementation][7]
is a library supporting distributed-memory parallel computing. It can be
used from many languages but it is most commonly used from C, C++, and
Fortran.

### OpenSHMEM

[OpenSHMEM](http://openshmem.org/) is a library for one-sided
communication.

# References

[1] https://en.wikipedia.org/wiki/Radix_sort
[2] https://en.wikipedia.org/wiki/Supercomputer
[3] https://en.wikipedia.org/wiki/Computer_cluster
[4] https://en.wikipedia.org/wiki/TOP500
[5] https://www.mpi-forum.org/
[6] https://www.open-mpi.org/
[7] https://www.mpich.org/
