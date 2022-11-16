#! /usr/bin/bash

# Find the openblas lib:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/users/spoutas/Libs/OpenBLAS-install/lib

# Configure
cmake -S . -B build/

# Build
cmake --build build/

# Run tests
ctest --test-dir build/ -V

