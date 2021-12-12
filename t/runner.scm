
(define-module (guile-gpu t runner)
  #:use-module (ice-9 match)
  #:use-module (ice-9 format)
  #:use-module (ffi blis arrays) ; from guile-ffi-cblas/mod
  #:use-module (ffi cblas)       ; from guile-ffi-cblas/mod
  #:use-module (guile-gpu common)
  #:use-module (guile-gpu gpu)

  #:use-module (guile-gpu sigmoid)
  #:use-module (guile-gpu mat)
  #:use-module (guile-gpu sigmoid)
  #:use-module (guile-gpu softmax)
  #:use-module (guile-gpu t test-common)

  #:export (tests-runner)
  #:re-export (test-env-set test-env?))

(define (tests-runner)

  (init-rand)

  ;;; Load ML/RL
  (sigmoid-init) ; FIX: passes even if uninitialized

  (set-current-module (resolve-module '(guile-gpu t runner)))
  (format #t "running tests in scheme-module: ~s~%" (current-module))
  ;;; tests
  (load "test-softmax.scm")
  (load "test-cblas.scm")
  (load "test-gpu.scm") ; test gpu-abstraction without hardware
  (load "test-gpu-blas.scm") ; test gpu-abstraction with hardware

  (run-tests '(;; reference
               test-softmax
               ;; FFI cblas
               test-cblas-copy
               test-cblas-sscal
               test-cblas-saxpy
               test-cblas-saxpy-2
               test-cblas-saxpy-3
               test-cblas-blas-sgemv
               ;; GPU blas
               test-gpu-blas-saxpy
               test-gpu-blas-sgemv
               test-gpu-array-copy
               test-gpu-sscal
               ;test-gpu-rocm-sigmoid ; FIX: need to move out NN stuff
               )))
