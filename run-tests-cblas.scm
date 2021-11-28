
(import (ice-9 format))
(import (system foreign) (ice-9 match))

(import (ffi cblas))

(load "common.scm")
(load "mat.scm")

(load "t/test-cblas-blas.scm")

(init-rand)
(test-cblas-blas-sgemv)
