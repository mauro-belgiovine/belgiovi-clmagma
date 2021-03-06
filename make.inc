#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.0.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of	Colorado, Denver
#      April 2012
#//////////////////////////////////////////////////////////////////////////////

# setenv AMD_CLBLAS_STORAGE_PATH /home/tomov/cl_magma
#
# GPU_TARGET specifies for which GPU you want to compile MAGMA:
#     "Tesla" (NVIDIA compute capability 1.x cards)
#     "Fermi" (NVIDIA compute capability 2.x cards)
#     "AMD"   (clMAGMA with AMD cards)
# See http://developer.nvidia.com/cuda-gpus
GPU_TARGET = Fermi

CC        = g++
NVCC      = nvcc
FORT      = ifort

ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib

OPTS      = -O0 -DADD_ -g -Wall
FOPTS     = -O0 -DADD_ -cpp 
#OPTS     = -O3 -DADD_ -g
#NVOPTS   = -O3 -DADD_ --compiler-options -fno-strict-aliasing -DUNIX -g -G
NVOPTS    = -O3 -DADD_ --compiler-options -fno-strict-aliasing -DUNIX
LDOPTS    = -fPIC -Xlinker -zmuldefs

# to set MKLROOT run: /mnt/scratch/sw/intel/2011.2.137/mkl/bin/mklvars.sh intel64
#MKLROOT   = /mnt/scratch/sw/intel/2011.2.137/composerxe-2011.2.137/mkl

#LIB       = -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -fopenmp
LIB        = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lm -fopenmp
LIB       += -lclAmdBlas -lOpenCL

GPUBLAS   = /home/GUEST/belgiovi/clAmdBlas-1.10.321

LIBDIR    = -L$(MKLROOT)/lib/intel64 \
            -L$(MKLROOT)/../compiler/lib/intel64 \
            -L$(GPUBLAS)/lib64
INC       = -I$(GPUBLAS)/include -I /software/cuda-5.0/include

