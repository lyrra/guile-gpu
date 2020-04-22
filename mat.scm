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
                    (* 0.01 (- 0.5 (random-uniform))))))
    (else (throw 'bad-array-type (array-type A))))
  A)

(define (array-zero! arr)
  (array-map! arr (lambda (x) 0) arr))

(define (array-inc! arr pos val)
  (let ((v (array-ref arr pos)))
    (array-set! arr (+ v val) pos)))

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

(define (assert-array-equal arra arrb)
  (let ((eps 0.001)) ; epsilon would be governed by (single,double) precision
    (array-for-each (lambda (a b)
                      (test-assert (> eps (abs (- a b)))
                              (format #f "[~f /= ~f] (epsilon: ~f)" a b eps)))
                    arra arrb)))

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
