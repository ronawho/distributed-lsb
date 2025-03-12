# HPX Implementation of LSD Radix Sort

`hpx_lsbsort.cpp` is a C++ and HPX implementation of LSD Radix Sort.

This version has scaling problems when run on an HPE Cray Supercomputing
EX system, so this README only describes how to run it locally. I was
seeing slow performance and hangs with the MPI and LCI parcelports. If
you are familiar with HPX and can help troubleshoot it, I would
appreciate any help!

# Building

First, it is necessary to get a copy of the PCG random number generator,
if it is not already available through a system-wide installation:

```
git clone https://github.com/imneme/pcg-cpp.git
```

Next, it's necessary to get a copy of HPX built and installed somewhere
you have access to. I used cmake like this:

```
git clone https://github.com/STEllAR-GROUP/hpx.git
git checkout tags/v1.10.0

cd hpx
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/mppf/sw/ -DHPX_WITH_PARCELPORT_MPI=True ..
make
make install
```

Make sure to adjust the installation prefix to something appropriate for you.
`-DHPX_WITH_PARCELPORT_MPI=True` activates multinode support.


Once HPX is built, the programs in this directory can be compiled with
`cmake` and `make`:

```
mkdir -p build
cmake .. -DHPX_DIR=/home/mppf/sw/lib/cmake/HPX -DCMAKE_BUILD_TYPE=Release
make
cd ..
```

(update the `HPX_DIR` above to point to the `cmake/HPX` directory
 wherever you installed HPX).


# Running

mpirun can launch it to run multiple ranks on the local system:

```
mpirun -n 4 ./build/hpx_lsbsort --n 100
```

