;
;
;
;(define-module (guile-gpu tests))

(import (guile-gpu common))
(import (guile-gpu gpu))
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
      ; do some module voodoo, because we need to do side-effect stuff in the correct package
      (let ((cur-mod (current-module)))
        (set-current-module (resolve-module '(guile-gpu gpu)))
        (load "rocm-blas.scm")
        ; for some reason, this is not needed. Though init-rocblas and init-rocblas-thread should be interned in the '(guile-gpu tests) obarray
        ; ((@@ (guile-gpu gpu) init-rocblas))
        ; ((@@ (guile-gpu gpu) init-rocblas-thread) 0)
        (init-rocblas)
        (init-rocblas-thread 0)
        (set-current-module cur-mod))))))

(tests-runner)
