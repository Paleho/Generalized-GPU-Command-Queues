#! /usr/bin/bash

# Configure
cmake -S . -B build/ -DBACKEND=$1

# Build
cmake --build build/

# Find the openblas lib:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/users/spoutas/Libs/OpenBLAS-install/lib

# Find the hipblas lib:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm-5.4.0/lib
