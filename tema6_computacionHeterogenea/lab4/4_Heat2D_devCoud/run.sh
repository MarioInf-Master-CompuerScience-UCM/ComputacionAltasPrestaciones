#!/bin/bash
export OMP_TARGET_OFFLOAD=DEFAULT

source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is running OMP_Offload Module1 -- Intro to OpenMP offload - 1 of 1 heat.c/f90
echo "########## Compiling"
icx -qopenmp -fopenmp-targets=spir64 -Wall -Wextra -std=c99 -O3 -xhost -qopt-report -fopenmp -g heat.c -o heat || exit $?
echo "########## Executing"
./heat
echo "########## Done"