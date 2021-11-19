export LD_LIBRARY_PATH=/opt/rocm/lib:.

# point out your blas-library: libblas.so, libatlas.so, /usr/lib/lapack/cygblas-0.dll etc
#export GUILE_FFI_CBLAS_LIBNAME=/usr/lib/libblas.so
#export GUILE_FFI_CBLAS_LIBNAME=/usr/lib/x86_64-linux-gnu/libopenblas.so
#export GUILE_FFI_CBLAS_LIBNAME=/opt/rocm/rocblas/lib/librocblas.so.0.1.30300

export GUILE_FFI_ROCM_LIBNAME=librocblas.so
export GUILE_FFI_NNGPU_LIBNAME=lib/gpu/libnn_gpu.so

#export GUILE_FFI_CBLAS_LIBNAME=librocmblas_shim.so

# path to guile-ffi-cblas git repository
#GUILE_CODE_LOAD_PATH=~/2020/git/guile-ffi-cblas/mod
#GUILE_CODE_LOAD_PATH="-L $HOME/pg -L $HOME/g"

#--------------------------------------------------------------------

# point out your blas-library: libblas.so, libatlas.so, /usr/lib/lapack/cygblas-0.dll etc
#export GUILE_FFI_CBLAS_LIBNAME=/usr/lib/libblas.so

# C modules used by td-gammon
#export GUILE_FFI_NNGPU_LIBNAME=lib/gpu/libnn_gpu.so
#export GUILE_FFI_NNPUBEVAL_LIBNAME=lib/pubeval/libnn_pubeval.so

# Path to external guile modules
#   - path to guile-ffi-cblas git repository (https://github.com/lloda/guile-ffi-cblas)
GUILE_CODE_LOAD_PATH="-L $HOME/pg/guile-ffi-cblas/mod -L $HOME/g"
#GUILE_CODE_LOAD_PATH="-L $HOME/pg"

