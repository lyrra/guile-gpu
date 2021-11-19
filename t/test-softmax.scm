(define-test (test-softmax)
  (let* ((rows 3)
         (rv (gpu-make-vector rows))
         (arr (gpu-array rv))
         (argmax 0))
    (array-map! arr (lambda (x) (random-uniform)) arr)
    (let ((m 0))
      (do ((i 0 (1+ i)))
          ((= i rows))
        (let ((x (array-ref arr i)))
          (if (> x m) 
            (begin
              (set! argmax i)
              (set! m x))))))
    (format #t "row: ~s~%" (gpu-array rv))
    (format #t "argmax: ~s~%" argmax)
    (softmax rv)
    (format #t "softmax: ~s~%" (gpu-array rv))
    (let ((m (array-ref arr argmax)))
      (do ((i 0 (1+ i)))
          ((= i rows))
        (test-assert (>= m (array-ref arr i)))))
    ; grad
    (let ((grad (grad-softmax rv 0)))
      (format #t "grad-softmax: ~s~%" grad)
      )))
