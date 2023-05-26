
(define-module (guile-gpu mat)
  #:use-module (ice-9 match)
  #:use-module (guile-gpu common)
  #:use-module (guile-gpu gpu)
  #:use-module (ffi cblas)       ; from guile-ffi-cblas/mod
  #:export (array-copy-bytevector
            array-matrix-scale!
            sv+!
            sv-!
            svvs*!
            ref-saxpy!
            ref-sgemv!))

(define (array-matrix-scale! a m)
  (let ((len (array-length m)))
    (do ((i 0 (1+ i))) ((= i len))
      (cblas-sscal! a (array-cell-ref m i)))))

; copy from array to bytevector
(define (array-copy-bytevector dst src)
  (match (array-dimensions dst)
    ((r c)
     (do ((i 0 (1+ i))) ((= i r))
     (do ((j 0 (1+ j))) ((= j c))
       (array-set! dst (f32vector-ref src (+ (* i c) j)) i j))))
    ((r)
     (do ((i 0 (1+ i))) ((= i r))
       (array-set! dst (f32vector-ref src i) i)))))

;;;;
;;;; blas reference
;;;;

(define (ref-saxpy! a x y)
  (match (array-dimensions x)
    ((r)
     (do ((j 0 (+ j 1))) ((= j r))
       (let* ((o (array-ref x j))
              (e (array-ref y j)))
         (array-set! y (+ e (* a o)) j))))))

; y := alpha*A*x + beta*y
(define (ref-sgemv! alpha A transA x beta y)
  (match (array-dimensions A)
    ((r c)
     (do ((i 0 (+ i 1))) ((= i r))
       (let ((s 0))
         (do ((j 0 (+ j 1))) ((= j c))
           (set! s (+ s (* alpha (array-ref A i j)
                                 (array-ref x j)))))
         (array-set! y (+ s (* beta (array-ref y i))) i))))))


; blas style, consider replacing these with calls into blas
(define (sv+! dst src1 src2)
  (array-map! dst (lambda (a b)
                    (+ a b))
              src1 src2))
(define (sv-! dst src1 src2)
  (array-map! dst (lambda (a b)
                    (- a b))
              src1 src2))

(define (svvs*! dst vec sc)
  (array-map! dst (lambda (v) (* v sc))
              vec))
