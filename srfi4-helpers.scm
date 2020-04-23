
(import (srfi srfi-4))

(define (srfi4-type-size srfi4-type)
  (case srfi4-type
    ((f32) 4)
    ((f64 c32) 8)
    ((c64) 16)
    (else (throw 'bad-srfi4-type-II srfi4-type))))

(define (srfi4-type->type srfi4-type)
  (case srfi4-type
    ((f32) float)
    ((f64) double)
    ((c32 c64) '*)
    (else (throw 'no-ffi-type-for-type srfi4-type))))

(define (srfi4-type->real srfi4-type)
  (case srfi4-type
    ((f32 c32) 'f32)
    ((f64 c64) 'f64)
    (else (throw 'no-real-type-for-type srfi4-type))))

(define (srfi4-type->real-type srfi4-type)
  (case srfi4-type
    ((f32 c32) float)
    ((f64 c64) double)
    (else (throw 'no-ffi-type-for-real-type srfi4-type))))
