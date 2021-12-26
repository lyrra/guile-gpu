
(define-module (guile-gpu mat)
  #:use-module (ice-9 match)
  #:use-module (guile-gpu common)
  #:use-module (guile-gpu gpu)
  #:use-module (ffi blis arrays) ; from guile-ffi-cblas/mod
  #:use-module (ffi cblas)       ; from guile-ffi-cblas/mod
  #:export (loop-array
            rand-v!
            rand-m!
            array-zero!
            array-copy
            array-scopy!
            array-inc!
            array-matrix-scale!
            sv+!
            sv-!
            svvs*!
            ref-saxpy!
            ref-sgemv!
            ref-sigmoid
            ref-sigmoid-grad))

; ---------------------------------
; constant times a vector plus a vector
; ---------------------------------
; y = a*x + y
; (saxpy! alpha vec vec)

; ---------------------------------
; scopy copy x into y
; ---------------------------------
; (scopy! Vsrc Vdst)

; ---------------------------------
; swap x and y
; (sswap! vec vec)

; ---------------------------------
; sgemv alpha A X beta Y
; ---------------------------------
; multiply matrix with vector
; alpha*sum_j(A_{ij} * X_j) + beta*Y_i -> Y_i

; -------------------------------------
; sgemm alpha M1 trans1 M2 trans2 beta Mdst
; -------------------------------------
; (sgemm! alpha A transA B transB beta C)

(import (srfi srfi-1) (srfi srfi-11))


(define (fill-v! A lst)
  (case (array-type A)
    ((f32 f64)
      (array-index-map! A (lambda (i)
                            (if (>= i (length lst))
                              1.0
                              (list-ref lst i)))))
    (else (throw 'bad-array-type (array-type A))))
  A)

(define (fill-m! A lst)
  (case (array-type A)
    ((f32 f64) (array-index-map!
                A (lambda (i j)
                    (if (>= i (length lst))
                      1.0
                      (if (>= j (length (list-ref lst i)))
                        1.0
                        (list-ref (list-ref lst i) j))))))
    (else (throw 'bad-array-type (array-type A))))
  A)

(define (rand-v! A)
  (case (array-type A)
    ((f32 f64)
      (array-index-map! A (lambda (i)
                            (* (- 0.5 (random-uniform)) 0.1))))
    (else (throw 'bad-array-type (array-type A))))
  A)

(define (rand-m! A)
  (case (array-type A)
    ((f32 f64) (array-index-map!
                A (lambda (i j)
                    (* 0.1 (- 0.5 (random-uniform))))))
    (else (throw 'bad-array-type (array-type A))))
  A)

(define (array-zero! arr)
  (array-map! arr (lambda (x) 0) arr))

(define (array-max arr)
  (let ((m #f))
    (array-for-each (lambda (x)
                      (if (or (not m) (> x m))
                          (set! m x)))
                    arr)
    m))

(define* (array-inc! arr pos #:optional v)
  (array-set! arr
              (+ (array-ref arr pos) (or v 1))
              pos))

(define (loop-array fun arr)
  (let ((i 0))
    (array-for-each (lambda (x)
                      (fun i x)
                      (set! i (+ i 1)))
                    arr)))

(define (array-copy src)
  (let ((dims (array-dimensions src)))
    (let ((dst (apply make-typed-array (append (list 'f32 *unspecified*) dims))))
      (array-map! dst (lambda (x) x) src)
      dst)))

(define (array-scopy! src dst)
  (array-map! dst (lambda (x) x) src))

(define (array-matrix-scale! a m)
  (let ((len (array-length m)))
    (do ((i 0 (1+ i))) ((= i len))
      (sscal! a (array-cell-ref m i)))))

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

;;; NN

(define (ref-sigmoid z)
  (/ 1. (+ 1. (exp (- z)))))

(define (ref-sigmoid-grad z)
  (let ((a (ref-sigmoid z)))
    (* a (- 1 a))))

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
