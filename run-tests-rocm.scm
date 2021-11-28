
(import (ice-9 format))
(import (system foreign) (ice-9 match))

(load "common.scm")
(load "mat.scm")
(load "srfi4-helpers.scm")
(load "rocm-blas.scm")

(load "t/test-rocm-blas.scm")

(init-rand)
(init-rocblas)
(init-rocblas-thread)

(test-rocm-blas-saxpy)
(test-rocm-blas-sgemv)

(quit-rocblas)
