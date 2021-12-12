# ROCM installation
export ROCM_PATH=/opt/rocm
export ROCM_PATH=$HOME/rocm4

export LD_LIBRARY_PATH=$ROCM_PATH/lib

# guile-ffi-cblas project uses the environment variables
# GUILE_FFI_CBLAS_LIBNAME and GUILE_FFI_BLIS_LIBNAME to
# search for its cblas and blis c-libraries.
# Use these to point out your installed blas-library (eg libblas.so, /usr/lib/x86_64-linux-gnu/libcblas.so, /usr/lib/lapack/cygblas-0.dll etc)
#export GUILE_FFI_CBLAS_LIBNAME=/usr/lib/libblas.so
#export GUILE_FFI_CBLAS_LIBNAME=/usr/lib/x86_64-linux-gnu/libopenblas.so
#export GUILE_FFI_CBLAS_LIBNAME=/opt/rocm/rocblas/lib/librocblas.so.0.1.30300

export GUILE_FFI_ROCM_LIBPATH=$ROCM_PATH/lib
export GUILE_FFI_ROCM_LIBNAME=librocblas.so

# machine-learning specialized gpu library
export GUILE_FFI_NNGPU_LIBPATH=.
export GUILE_FFI_NNGPU_LIBNAME=lib/gpu/libnn_gpu.so

# Path to external guile modules
#   - path to guile-ffi-cblas git repository (https://github.com/lloda/guile-ffi-cblas)
GUILE_CODE_LOAD_PATH="-L $HOME/pg/guile-ffi-cblas/mod -L $HOME/g"

