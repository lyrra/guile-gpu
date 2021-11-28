=========
Guile-GPU
=========

What is it?
===========
 * A thin wrapper for A SUBSET of low-level matrix operations
   and common machine-learning routines.
 * BLAS FFI wrapper
   Gives access to hardware acceleration
   by wrapping either a GPU CBLAS library
   or a CPU CBLAS library.
 * AMD/ROCM BLAS library supported
 * CBLAS Open blas (libatlas) library supported
 
GPU BLAS
=========
  The GPU BLAS wrapper works against the usual BLAS
  specification from netlib's LINPACK.
  Ie FORTRAIN datastructures, which are column-major.
  Matrices are mapped into Guile/SCHEME row-major.
  AMD ROCM blas library are tested to work.

CPU BLAS
=========
  The CPU CBLAS wrapper can call any
  CBLAS compatible library. Again the reference
  implementation comes from netlib's LAPACK.
  Open CLBAS (libatlas) provides a CPU-accelerated
  implementation that works.

BLAS subset implemented
=======================
These are notable api functions, and blas operations supported:

* gpu-make-vector (rows) -- makes a vector (that can be HW-accel)

* gpu-make-matrix (rows cols) -- makes a matrice (that can be HW-accel)

* gpu-array-copy (rv src) -- copy pure guile array contents into an HW-accel array or vector

* gpu-array-sigmoid (src dst) -- sigmoid HW-accel

* gpu-sscal! (alpha x) -- blas SSCAL

* gpu-saxpy! (alpha x y #:optional rox roy) -- blas SAXPY

* gpu-sgemv! (alpha A transA x beta y) -- blas SGEMV

Common machine-learning routines
================================
 * sigmoid / gradient-sigmoid
   The scheme version uses a somewhat naive lookup-table.
   Whereas the GPU accelerated version will calculate
   sigmoid over an vector (parallel).
   GPU tuning of the cpp program may be necessary.

 * softmax / gradient-softmax
   Only a scheme reference implementation so far.

