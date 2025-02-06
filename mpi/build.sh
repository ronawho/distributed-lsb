#!/bin/bash

mpic++ -O3 mpi_lsbsort.cpp -o mpi_lsbsort -I pcg-cpp/include/

#mpic++ app.cpp && mpirun -n 2 ./a.out
