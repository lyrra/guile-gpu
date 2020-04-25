(define gpu-init-fun #f)
(define gpu-init-thread-fun #f)

(define (gpu-init)
  (if gpu-init-fun (gpu-init-fun)))

(define (gpu-init-thread threadno)
  (if gpu-init-thread-fun
    (gpu-init-thread-fun threadno)))

;;; device/host vector/matrix mappings

(define (gpu-make-vector rows)
  (make-struct/no-tail (make-vtable "pwpwpwpwpw")
                       (make-typed-array 'f32 *unspecified* rows)
                       0 ; dirty
                       0 ; type (vector/matrix)
                       #f ; c-pointer address
                       rows)) ; array length

(define (gpu-make-matrix rows cols)
  (make-struct/no-tail (make-vtable "pwpwpwpwpwpw")
                       ;(make-typed-array 'f32 *unspecified* (* rows cols))
                       (make-typed-array 'f32 *unspecified* rows cols)
                       0 ; dirty
                       1 ; type (vector/matrix)
                       #f ; c-pointer address
                       rows
                       cols))

(define (gpu-array  rv) (struct-ref rv 0))
(define (gpu-dirty  rv) (struct-ref rv 1))
(define (gpu-dirty-set! rv val) (struct-set! rv 1 val))
(define (gpu-type   rv) (struct-ref rv 2))
(define (gpu-addr   rv) (struct-ref rv 3))
(define (gpu-rows   rv) (struct-ref rv 4))
(define (gpu-cols   rv) (struct-ref rv 5))

; if a gpu (device) is used, we have both host and device memory
(define (gpu-load-array rv) #f)
(define (gpu-save-array rv) #f)
(define (gpu-refresh rv) #f)
(define (gpu-refresh-host rv) #f)
(define (gpu-refresh-device rv) #f)

(define (gpu-array-dimensions rv)
  (if (= (gpu-type rv) 0)
      (list (gpu-rows rv))
      (list (gpu-rows rv) (gpu-cols rv))))

(define (gpu-array-apply rv fun)
  (gpu-refresh rv)
  (let ((bv (gpu-array rv)))
    (array-map! bv fun bv)
    (gpu-dirty-set! rv 1)))

(define (gpu-array-map! dst fun src)
  (gpu-refresh dst)
  (gpu-refresh src)
  (let ((dv (gpu-array dst))
        (sv (gpu-array src)))
    (array-map! dv fun sv)
    (gpu-dirty-set! dst 1)))

(define (gpu-array-for-each fun rvs)
  (let ((arrs (map (lambda (rv)
                     ;(format #t "gpu-array-for-each rv: ~s~%" rv)
                     (gpu-refresh-host rv)
                     (gpu-array rv)) rvs)))
    (apply array-for-each fun arrs)))

(define (gpu-array-copy rv src)
  (let* ((bv (gpu-array rv))
         (rows (gpu-rows rv)))
    (match (gpu-type rv)
     (0 ; vector
      (assert (= rows (array-length src)))
      (do ((i 0 (1+ i))) ((= i rows))
        (f32vector-set! bv i (array-ref src i))))
     (1
      (let ((cols (gpu-cols rv))
            (n 0))
        (assert (= (* cols rows)
                   (* (car (array-dimensions src)) (cadr (array-dimensions src)))))
        (do ((i 0 (1+ i))) ((= i rows))
        (do ((j 0 (1+ j))) ((= j cols))
          (array-set! bv (array-ref src i j) i j)
          ;(set! n (1+ n))
          )))))
    (gpu-dirty-set! rv 1)))


;;;; BLAS wrappers

(define* (gpu-saxpy! alpha x y #:optional rox roy)
  (cond
   ((or rox roy) ; row-offset
    (saxpy! alpha (if rox (array-cell-ref (gpu-array x) rox)
                      (gpu-array x))
                  (if roy (array-cell-ref (gpu-array y) roy)
                      (gpu-array y))))
   (else
    (saxpy! alpha (gpu-array x) (gpu-array y)))))

; default to using cpu-blas
(define (gpu-sgemv! alpha A transA x beta y)
  ;(sgemv! alpha A (if transA CblasTrans CblasNoTrans) x beta y)
  ; rocm
  (sgemv! alpha (gpu-array A) (if transA CblasTrans CblasNoTrans) (gpu-array x) beta (gpu-array y)))
