;
;
;
;(define-module (guile-gpu tests))

(import (guile-gpu t runner))

(begin
  ;;; check if gpu is used
  (let ((gpu #f))
    (do ((args (command-line) (cdr args)))
        ((eq? args '()))
      (if (string=? (car args) "--gpu")
          (set! gpu #t)))
    (cond
     (gpu
      (load "rocm-blas.scm")
      (init-rocblas)
      (init-rocblas-thread 0)))))

(tests-runner)
