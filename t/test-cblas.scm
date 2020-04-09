
(define (test-copy fun dst src ref n)
  ; set source, reference
  ; and clear destination
  (do ((i 0 (+ i 1)))
      ((= i n))
    (let ((x (random 256)))
      (array-set! src x i)
      (array-set! ref x i)
      (array-set! dst 0 i)))
  ; do copy
  (fun src dst)
  ;(display src) (display dst)
  ; ensure destination matches reference, and source is intact
  (array-for-each (lambda (d r)
                    (test-assert (= d r)
                                 (format #f "array-copy reference array mismatch")))
                  dst ref))

(define (test-blas-copy)
  (let* ((n (+ 1 (random 129)))
         (svec (make-typed-array 'f32 *unspecified* n))
         (sref (make-typed-array 'f32 *unspecified* n))
         (dvec (make-typed-array 'f32 *unspecified* n)))
    ; copy array using array-map!
    (test-copy (lambda (s d)
                 (array-map! d (lambda (x) x) s))
               dvec svec sref n)
    ; copy array using array-copy
    (test-copy array-copy! dvec svec sref n)
    ; copy array using C-BLAS
    (test-copy scopy! dvec svec sref n)))
