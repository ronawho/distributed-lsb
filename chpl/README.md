# Chapel Implementation of LSD Radix Sort

`arkouda-radix-sort.chpl` is a [Chapel](https://chapel-lang.org/)
implementation of LSD Radix Sort. It is a standalone version of the
LSD Radix sort developed as part of the
[Arkouda](https://github.com/Bears-R-Us/arkouda) project.

# Building

```
chpl --fast arkouda-radix-sort.chpl
```

# Running

To run on 4 nodes (including possibly simulating 4 nodes locally with
https://chapel-lang.org/docs/platforms/udp.html#using-udp):

```
./arkouda-radix-sort -nl 4
```

Use `--n` to specify the problem size, e.g.:
```
./arkouda-radix-sort -nl 4 --n=100
```

# Details of Measured Version

Performance was measured on an HPE Cray Supercomputing EX using 64 nodes
(each using 128 cores) and a problem size of 68719476736 elements total
(so the elements require 16 GiB of space per node).

`chpl` version was `2.4.0 pre-release (a0bf1696e4)`

Compile command:

```
chpl --fast arkouda-radix-sort.chpl
```

Run command:

```
./arkouda-radix-sort -nl 64 --n=68719476736
```

Output:

```
Generating 68719476736 2*uint(64) elements
Sorting
Sorted 68719476736 elements in 10.533 s
That's 6524.23 M elements sorted / s
```

Further details:
 * used the bundled LLVM backend with LLVM 19 (`export CHPL_LLVM=bundled`)
 * used `CHPL_COMM=ofi` and `CHPL_LIBFABRIC=system`
   (the defaults on this system)
 * loaded modules `cray-pmi`, `gcc-native`, and  `PrgEnv-gnu` in addition
   to the system defaults
 * used the system default processor selection module `craype-x86-rome`
 * used these additional environment variables
   ```
   export CHPL_LAUNCHER_MEM=unset
   export CHPL_RT_MAX_HEAP_SIZE="50%"
   export SLURM_UNBUFFEREDIO=1
   ```
