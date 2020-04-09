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
