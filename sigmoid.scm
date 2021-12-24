(define-module (guile-gpu sigmoid)
  #:export (sigmoid-init
            sigmoid-grad
            array-sigmoid))

(define *sigmoid-table* #f)

(define (sigmoid-real z) (/ 1. (+ 1. (exp (- z)))))

(define (sigmoid x)
  (let ((i (inexact->exact (truncate (+ (* (/ x 40) 65536) 32768)))))
    (if (< i 0) (set! i 0))
    (if (> i 65535) (set! i 65535))
    (array-ref *sigmoid-table* i)))

(define (sigmoid-grad z)
  (let ((a (sigmoid z)))
    (* a (- 1 a))))

; Dsigmoid(x) = sigmoid(x) (1 - sigmoid(x))
(define (array-sigmoid src dst)
  (array-map! dst (lambda (z) (sigmoid z))
                  src))

; calculate gradient GRAD(weight, output)
(define (set-sigmoid-gradient! grad netz)
  (array-map! grad (lambda (z) (sigmoid-grad z))
              netz))

(define (sigmoid-init)
  (set! *sigmoid-table*
         (make-typed-array 'f32 *unspecified* 65536))
  (do ((i 0 (+ i 1)))
      ((= i 65536))
    ;(if (or (< i 10) (> i 65526)) (format #t "~a: ~f~%" i (- (* 40 i (/ 1 65536)) 20)))
    (array-set! *sigmoid-table*
                (sigmoid-real (- (* 40 i (/ 1 65536)) 20))
                i)))

