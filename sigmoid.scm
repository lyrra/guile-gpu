(define-module (guile-gpu sigmoid)
  #:export (sigmoid-init
            ref-sigmoid-real
            ref-sigmoid
            ref-sigmoid-grad
            array-sigmoid))

(define *sigmoid-table* #f)

(define (ref-sigmoid-real z) (/ 1. (+ 1. (exp (- z)))))

(define (ref-sigmoid x)
  (let ((i (inexact->exact (truncate (+ (* (/ x 40) 65536) 32768)))))
    (if (< i 0) (set! i 0))
    (if (> i 65535) (set! i 65535))
    (array-ref *sigmoid-table* i)))

(define (ref-sigmoid-grad z)
  (let ((a (ref-sigmoid z)))
    (* a (- 1 a))))

; Dsigmoid(x) = sigmoid(x) (1 - sigmoid(x))
(define (array-sigmoid src dst)
  (array-map! dst (lambda (z) (ref-sigmoid z))
                  src))

; calculate gradient GRAD(weight, output)
(define (set-sigmoid-gradient! grad netz)
  (array-map! grad (lambda (z) (ref-sigmoid-grad z))
              netz))

(define (sigmoid-init)
  (set! *sigmoid-table*
         (make-typed-array 'f32 *unspecified* 65536))
  (do ((i 0 (+ i 1)))
      ((= i 65536))
    ;(if (or (< i 10) (> i 65526)) (format #t "~a: ~f~%" i (- (* 40 i (/ 1 65536)) 20)))
    (array-set! *sigmoid-table*
                (ref-sigmoid-real (- (* 40 i (/ 1 65536)) 20))
                i)))

