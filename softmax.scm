
(define-module (guile-gpu softmax)
  #:export (softmax
            grad-softmax))
(import (guile-gpu gpu))

(define (softmax rv)
  (gpu-refresh-host rv)
  (let* ((arr (gpu-array rv))
         (sum 0))
    (array-map! arr exp arr)
    (array-for-each (lambda (x)
                      (set! sum (+ sum x)))
                    arr)
    (array-map! arr (lambda (x) (/ x sum)) arr)
    (gpu-dirty-set! rv 1)))

(define (grad-softmax rv j)
  (gpu-refresh-host rv)
  (let* ((rows (gpu-rows rv))
         (grad (make-typed-array 'f32 *unspecified* rows))
         (arr (gpu-array rv))
         (Sj (array-ref arr j)))
    (do ((i 0 (1+ i)))
        ((= i rows))
      (let ((x (array-ref arr i)))
        (array-set! grad
                    (if (= i j)
                      (* Sj (- 1 x)) ; Sj - Sj * Sj
                      (- (* Sj x)))  ; 0 - Sj * Sj
                    i)))
    grad))
