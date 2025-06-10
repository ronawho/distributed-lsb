# Rust + Lamellar Implementation of LSD Radix Sort

`lamellar_lsbsort` is a Rust and Lamellar implementation of LSD Radix
Sort.

*This version is still under development*

[Lamellar](https://github.com/pnnl/lamellar-runtime) is a Rust library
that supports High Performance Computing with a focus on PGAS approaches.
It supports both local threads and distributed-memory processes. 

# Building

(First, you will need to install Rust if you have not already.)

Lamellar is the main dependency here, although it has quite a few
dependencies in turn. Cargo can download and build all of these
dependences.

```
cd lamellar_lsbsort
cargo build --release
```

You can leave out `--release` if you would like a debug build.

# Running

You can run `lamellar_lsbsort` through cargo to check that it is working
at all.

```
cd lamellar_lsbsort
cargo run --release
```

However, that method of running it won't allow for multiple processes. To
run with multiple processes, use `lamellar_run.sh` (available at
https://raw.githubusercontent.com/pnnl/lamellar-runtime/refs/heads/master/lamellar_run.sh
).

For example, the commands below will run it locally with 4 processes each
with 8 threads:

```
cd lamellar_lsbsort
lamellar_run.sh -N=4 -T=8 target/release/lamellar_lsbsort
```

# HPC System

`spack install rust`
`spack load rust`

load PrgEnv-gnu (rust dependencies don't build with PrgEnv-cray loaded)
