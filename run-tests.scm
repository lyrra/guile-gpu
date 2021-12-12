;
;
;
;(define-module (guile-gpu tests))

(import (guile-gpu common))
(import (guile-gpu gpu))
(import (guile-gpu t runner))

(begin
  ;;; check if gpu is used
  (do ((args (command-line) (cdr args)))
      ((eq? args '()))
    (if (string=? (car args) "--gpu")
      (test-env-set #:gpu #t)))
  (cond
   ((test-env? #:gpu)
    (load "rocm-blas.scm")
    (gpu-init)
    (gpu-init-thread 0))))

(format #t "GPU type: ~s~%" (gpu-host))
(tests-runner)
