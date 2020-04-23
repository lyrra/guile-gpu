
(import (rnrs bytevectors))
(import (system foreign))

(eval-when (compile load eval)
  (define *dynlibfile*
    (dynamic-link (let ((lpath (getenv "GUILE_FFI_ROCM_LIBPATH"))
                        (lname (or (getenv "GUILE_FFI_ROCM_LIBNAME") "librocm")))
                    (if (and lpath (not (string=? lpath "")))
                        (string-append lpath file-name-separator-string lname)
                        lname)))))

(define (pointer-to-first A)
  (bytevector->pointer A ;(shared-array-root A)
                         ;(* (shared-array-offset A) (srfi4-type-size (array-type A)))
                         ))

(define (scalar->arg srfi4-type a)
  (case srfi4-type
    ((f32 f64) a)
    ((c32 c64) (bytevector->pointer (shared-array-root (make-typed-array srfi4-type a))))
    (else (throw 'bad-array-type srfi4-type))))

;##################################################################

;;;; HIP

(define (hip-malloc numfloats)
  (let* ((bv (make-bytevector 8))
         (ptr (bytevector->pointer bv)))
    (let ((fun
      (pointer->procedure
       int (dynamic-func "hipMalloc" *dynlibfile*)
       (list '*     ; pointer to store address to allocated device memory
             int    ; number of floats to allocate
             ))))
      (if (not (= (fun ptr (* numfloats 4)) 0))
        (error "FAIL:hipMalloc"))
      (make-pointer (bytevector-u64-ref bv 0 (endianness little))))))

(define (hip-free dx)
  ((pointer->procedure
    int (dynamic-func "hipFree" *dynlibfile*)
    (list '*))     ; pointer to allocated device memory
   dx))


(define (hip-memcpy-to-device dst src)
  (let* ((len (bytevector-length src)))
    ; copy array to bytevector
    ((pointer->procedure
       int (dynamic-func "hipMemcpy" *dynlibfile*)
       (list '*     ; pointer to device memory
             '*     ; pointer to host memory
             int    ; number of bytes to move
             int)) ; host/device direction to move bytes in
     dst (bytevector->pointer src) len 1))) ; 1 = hipMemcpyHostToDevice

(define (hip-memcpy-from-device dst src len)
  ((pointer->procedure
      int (dynamic-func "hipMemcpy" *dynlibfile*)
      (list '*     ; pointer to host memory
            '*     ; pointer to device memory
            int    ; number of bytes to move
            int)) ; host/device direction to move bytes in
   ; args to hipMemcpy:
   (bytevector->pointer dst) src (* len 4) 2)) ; 2 = hipMemcpyDeviceToHost

;;;; ROCBLAS


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
                       (make-typed-array 'f32 *unspecified* (* rows cols))
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

(define (gpu-free-array rv)
  (let* ((addr (gpu-addr rv)))
    (if addr (hip-free addr))))

; load an array onto GPU/device
(define (gpu-load-array rv)
  (let* ((bv   (gpu-array rv))
         (addr (gpu-addr  rv))
         (rows (gpu-rows rv))
         (len (if (= (gpu-type rv) 0)
                  rows
                  (* (gpu-cols rv) rows))))
    (if (not addr) ; no memory on device
      (begin
        (struct-set! rv 3 (hip-malloc len))
        (set! addr (gpu-addr rv))))
    (hip-memcpy-to-device addr bv)
    (gpu-dirty-set! rv 0)))

; save into the rocm object from array at GPU/device
(define (gpu-save-array rv)
  (let* ((bv   (gpu-array rv))
         (addr (gpu-addr rv))
         (rows (gpu-rows rv))
         (len (if (= (gpu-type rv) 0)
                  rows
                  (* (gpu-cols rv) rows))))
    (if (not addr) ; no memory on device, user-error
      (error "expected allocated memory-array on device"))
    (hip-memcpy-from-device bv addr len)
    (gpu-dirty-set! rv 0)))

(define (gpu-refresh rv)
  (match (gpu-dirty rv)
    (0 #t) ; not-dirty
    (1 (gpu-load-array rv))
    (2 (gpu-save-array rv)))
  (gpu-dirty-set! rv 0))

(define (gpu-refresh-device rv)
  (if (= (gpu-dirty rv) 1)
    (begin
      (gpu-load-array rv)
      (gpu-dirty-set! rv 0))))

(define (gpu-array-apply rv fun)
  (let ((bv (gpu-array rv)))
    (array-map! bv fun bv)
    (gpu-dirty-set! rv 1)))

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
          (f32vector-set! bv n (array-ref src i j))
          (set! n (1+ n)))))))
    (gpu-dirty-set! rv 1)))

;;;; rocblas API (and some HIP functions)

(define %rocblas-handle #f)
(define %rocblas-v1 #f)

(define (init-rocblas)
  (set! %rocblas-handle (make-parameter #f))
  (let ((bv (make-bytevector 8)))
    ((pointer->procedure
              int (dynamic-func "rocblas_create_handle" *dynlibfile*)
              (list '*     ; rocblas_handle handle
                    ))
      (bytevector->pointer bv))
    (%rocblas-handle (make-pointer (bytevector-u64-ref bv 0 (endianness little))))
    ;--------------------
    ;rocblas_status rocblas_set_pointer_mode(rocblas_handle handle, rocblas_pointer_mode pointer_mode)
    ((pointer->procedure
              int (dynamic-func "rocblas_set_pointer_mode" *dynlibfile*)
              (list '*     ; rocblas_handle handle
                    int))
     (%rocblas-handle)
     0) ; set pointer-mode to host
    ; setup 1-element-array (for cache)
    (set! %rocblas-v1 (make-parameter #f))))

(define (quit-rocblas)
  ;rocblas_status rocblas_destroy_handle(rocblas_handle handle)
  ((pointer->procedure
            int (dynamic-func "rocblas_destroy_handle" *dynlibfile*)
            (list '*))     ; rocblas_handle handle
     (%rocblas-handle)))

(define (init-rocblas-thread)
  (%rocblas-v1 (gpu-make-vector 1))) ; single-element vector

; get the single-element-vector value
(define (gpu-get-v1)
  (let ((rv (%rocblas-v1)))
    (gpu-refresh rv)
    (f32vector-ref (gpu-array rv) 0)))

;;;; rocblas

(define (rocblas-result-assert res)
  (if (not (= res 0))
      (error "rocblas result fail")))

;-----------------------------------------------------------------
;
; Level-1 BLAS
;
;-----------------------------------------------------------------

; -----------------------------
; copy x y
;
; rocblas_status
; rocblas_scopy(rocblas_handle handle,
;               rocblas_int n,
;               const float *x, rocblas_int incx,
;               float *y,       rocblas_int incy)

(begin
  (define _rocblas_scopy
    (pointer->procedure
    int (dynamic-func "rocblas_scopy" *dynlibfile*)
     (list '*     ; rocblas_handle handle
           int    ; rocblas_int n
           '*     ; const float *x
           int    ; rocblas_int incx
           '*     ; float *y
           int))) ; rocblas_int incy
  (define (scopy! x y)
    (gpu-refresh-device x)
    (gpu-dirty-set! y 2)
    (rocblas-result-assert
     (_rocblas_scopy (%rocblas-handle)
                     (gpu-rows x)
                     (gpu-addr x) 1 ; 1=stride
                     (gpu-addr y) 1))))

; -----------------------------
; swap x y
;
; rocblas_status
; rocblas_sswap(rocblas_handle handle,
;               rocblas_int n,
;               float *x, rocblas_int incx,
;               float *y, rocblas_int incy)

(begin
  (define _rocblas_sswap
    (pointer->procedure
     int (dynamic-func "rocblas_sswap" *dynlibfile*)
     (list '*     ; rocblas_handle handle
           int    ; rocblas_int n
           '*     ; const float *x
           int    ; rocblas_int incx
           '*     ; float *y
           int))) ; rocblas_int incy
  (define (sswap! x y)
    (gpu-refresh-device x)
    (gpu-refresh-device y)
    (gpu-dirty-set! x 2)
    (gpu-dirty-set! y 2)
    (rocblas-result-assert
     (_rocblas_sswap (%rocblas-handle)
                     (gpu-rows x)
                     (gpu-addr x) 1
                     (gpu-addr y) 1))))

; -----------------------------
; axpy
;
; rocblas_status
; rocblas_saxpy(rocblas_handle handle,
;               rocblas_int n,
;               const float *alpha,
;               const float *x, rocblas_int incx,
;               float *y, rocblas_int incy)

(begin
  (define _rocblas_saxpy
    (pointer->procedure
     int (dynamic-func "rocblas_saxpy" *dynlibfile*)
     (list '*     ; rocblas_handle handle
           int    ; rocblas_int n
           '*     ; const float *alpha
           '*     ; const float *x
           int    ; rocblas_int incx
           '*     ; float *y
           int))) ; rocblas_int incy
  (define (saxpy! a x y)
    (gpu-refresh-device x)
    (gpu-refresh-device y)
    (gpu-dirty-set! y 2)
    (rocblas-result-assert
     (_rocblas_saxpy (%rocblas-handle)
                     (gpu-rows x)
                     (scalar->arg 'c32 a)
                     (gpu-addr x) 1
                     (gpu-addr y) 1))))


; -----------------------------
; scal
;
; rocblas_status
; rocblas_sscal (rocblas_handle handle,
;                rocblas_int n,
;                const float *alpha,
;                float *x, rocblas_int incx)

(begin
  (define _rocblas_sscal
    (pointer->procedure
     int (dynamic-func "rocblas_sscal" *dynlibfile*)
     (list '*     ; rocblas_handle handle
           int    ; rocblas_int n
           '*     ; const float *alpha
           '*     ; float *x
           int))) ; rocblas_int incx
  (define (sscal! a x)
    (gpu-refresh-device x)
    (gpu-dirty-set! x 2)
    (rocblas-result-assert
     (_rocblas_sscal (%rocblas-handle)
                     (gpu-rows x)
                     (scalar->arg 'f32 a)
                     (gpu-addr x) 1))))

(begin
  (define _rocblas_sdot
    (pointer->procedure
     int  (dynamic-func "rocblas_sdot" *dynlibfile*)
     (list '*     ; rocblas_handle handle
           int    ; rocblas_int n
           '*     ; float *x
           int    ; rocblas_int incx
           '*     ; float *y
           int    ; rocblas_int incy
           '*)))  ; float *result
  (define (sdot! x y r)
    (gpu-refresh-device x)
    (gpu-refresh-device y)
    (gpu-refresh-device r)
    (gpu-dirty-set! r 2)
    (rocblas-result-assert
     (_rocblas_sdot (%rocblas-handle)
                    (gpu-rows x)
                    (gpu-addr x) 1
                    (gpu-addr y) 1
                    (gpu-addr r)))))

; -----------------------------
; isamax
;
; rocblas_status
; rocblas_isamax(rocblas_handle handle,
;                rocblas_int n,
;                const float *x, rocblas_int incx,
;                rocblas_int *result)

(begin
  (define _rocblas_isamax
    (pointer->procedure
     int  (dynamic-func "rocblas_isamax" *dynlibfile*)
     (list '*     ; rocblas_handle handle
           int    ; rocblas_int n
           '*     ; float *x
           int    ; rocblas_int incx
           '*)))  ; rocblas_int *result
  (define (isamax x)
    (gpu-refresh-device x)
    (gpu-refresh-device (%rocblas-v1))
    (rocblas-result-assert
     (_rocblas_isamax (%rocblas-handle)
                      (gpu-rows x)
                      (gpu-addr x) 1
                      (gpu-addr (%rocblas-v1))))
    (gpu-get-v1)))

;-----------------------------------------------------------------
;
; Level-2 BLAS
;
;-----------------------------------------------------------------

; -----------------------------
; sgemv alpha*A*x + beta*y -> y
; rocblas_status
; rocblas_sgemv(rocblas_handle handle,
;               rocblas_operation trans,
;               rocblas_int m, rocblas_int n,
;               const float *alpha,
;               const float *A, rocblas_int lda,
;               const float *x, rocblas_int incx,
;               const float *beta,
;               float *y, rocblas_int incy)

(begin
  (define _rocblas_sgemv
    (pointer->procedure
     int (dynamic-func "rocblas_sgemv" *dynlibfile*)
     (list '*        ; rocblas_handle handle
           int       ; rocblas_operation trans,
           int int   ; rocblas_int m, rocblas_int n,
           '*        ; const float *alpha,
           '* int    ; const float *A, rocblas_int lda,
           '* int    ; const float *x, rocblas_int incx,
           '*        ; const float *beta,
           '* int))) ; float *y, rocblas_int incy
  (define (sgemv! alpha A TransA x beta y)
    (gpu-refresh-device A)
    (gpu-refresh-device x)
    (gpu-refresh-device y)
    (gpu-dirty-set! y 2)
    (let ((M (gpu-rows A))
          (N (gpu-cols A)))
      (unless (= M (gpu-rows y)) (throw 'mismatched-Ay N (gpu-rows y)))
      (unless (= N (gpu-rows x)) (throw 'mismatched-Ax N (gpu-rows x)))
      (rocblas-result-assert
       (_rocblas_sgemv (%rocblas-handle)
                       (if TransA 112 111) ; convert to row-major+transpose
                       M N
                       (scalar->arg 'c32 alpha)
                       (gpu-addr A) M
                       (gpu-addr x) 1
                       (scalar->arg 'c32 beta)
                       (gpu-addr y) 1)))))
