
(define-test (test-gpu-blas-saxpy)
  (loop-subtests (tn)
  (let* ((rows (logand #xfffe (inexact->exact (truncate (+ 2 (* 512 (random-uniform)))))))
         (a 0.01)
         (rx (gpu-make-vector rows))
         (ry (gpu-make-vector rows)))
    (L 2 "saxpy rows: ~a~%" rows)
    (gpu-array-apply rx (lambda (x) (* 20 (- (random-uniform) 0.5))))
    (gpu-array-apply ry (lambda (x) (* 20 (- (random-uniform) 0.5))))
    (let ((x2 (array-copy (gpu-array rx)))
          (y2 (array-copy (gpu-array ry))))
      ;(gpu-load-array rx)
      ;(gpu-load-array ry)
      (gpu-saxpy! a rx ry)
      (gpu-refresh-host rx)
      (gpu-refresh-host ry)
      ;(gpu-save-array rx)
      ;(gpu-save-array ry)
      (ref-saxpy! a x2 y2)
      (assert-array-equal (gpu-array ry) y2))
    (gpu-free-array rx)
    (gpu-free-array ry))))


(define-test (test-gpu-blas-sgemv)
  (loop-subtests (tn)
  (let* ((rows (logand #xfffe (inexact->exact (truncate (+ 2 (* 8 (random-uniform)))))))
         (cols (logand #xfffe (inexact->exact (truncate (+ 2 (* 256 (random-uniform)))))))
         (alpha 0.01)
         (beta 0.5)
         (m2 (rand-m! (make-typed-array 'f32 *unspecified* rows cols)))
         (rm (gpu-make-matrix rows cols))
         (rx (gpu-make-vector cols))
         (ry (gpu-make-vector rows)))
      (gpu-array-copy rm m2)
      (gpu-array-apply rx (lambda (x) (- (random-uniform) .5)))
      (gpu-array-apply ry (lambda (x) (- (random-uniform) .5)))
      (gpu-load-array rm)
      (gpu-load-array rx)
      (gpu-load-array ry)
      (let ((x2 (array-copy (gpu-array rx)))
            (y2 (array-copy (gpu-array ry))))
        ; rocm-blas
        (gpu-sgemv! alpha rm #f rx beta ry)
        (gpu-save-array ry)
        ; cblas-blas
        ;(sgemv! alpha (gpu-array rm) CblasNoTrans (gpu-array rx) beta (gpu-array ry))
        ; reference-blas
        (ref-sgemv! alpha m2 #f x2 beta y2)
        (let ((err 0.))
          (array-for-each (lambda (a b)
                            (set! err (+ err (abs (- a b)))))
                          (gpu-array ry) y2)
          (L 2 "rocm-sgemv dims: ~a, ~a err: ~a~%" rows cols err))
        (assert-array-equal (gpu-array ry) y2 (max .003 (* rows cols .00002))))
      (gpu-free-array rm)
      (gpu-free-array rx)
      (gpu-free-array ry))))
