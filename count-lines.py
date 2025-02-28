#!/usr/bin/env python3

import sys

def count_lines_ignoring_blank(file_path):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() != "":
                count += 1
    
    return count

if __name__ == "__main__":
    paths = ["chpl/arkouda-radix-sort-terse.chpl",
             "mpi/mpi_lsbsort-terse.cpp",
             "shmem/shmem_lsbsort-terse.cpp"]

    for p in paths:
        count = count_lines_ignoring_blank(p)
        print(f"{p:40} {count}")
