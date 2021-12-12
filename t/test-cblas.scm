; these tests works directly on an FFI'ed blas,
; such as when loading guile-ffi-cblas

(define (_test-copy fun dst src ref n)
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

(define-test (test-cblas-copy)
  (loop-subtests (i)
  (let* ((n (+ 1 (random (1+ i))))
         (svec (make-typed-array 'f32 *unspecified* n))
         (sref (make-typed-array 'f32 *unspecified* n))
         (dvec (make-typed-array 'f32 *unspecified* n)))
    ; copy array using array-map!
    (_test-copy (lambda (s d)
                 (array-map! d (lambda (x) x) s))
               dvec svec sref n)
    ; copy array using array-copy
    (_test-copy array-copy! dvec svec sref n)
    ; copy array using C-BLAS
    (_test-copy scopy! dvec svec sref n))))

(define-test (test-cblas-sscal)
  (loop-subtests (i)
  (let* (; reference arrays, computed by axiomal ref-saxpy
         (v (rand-v! (make-typed-array 'f32 0. (1+ i))))
         (m (rand-m! (make-typed-array 'f32 0. (1+ i) 32)))
         ; copy used by guile-cblas
         (v2 (array-copy v))
         (m2 (array-copy m)))
    ; scale vector
    (array-map! v (lambda (e) (* e 0.1)) v)
    (sscal! 0.1 v2)
    (assert-array-equal v v2)
    ; scale matrix
    (array-map! m (lambda (e) (* e 0.1)) m)
    (array-matrix-scale! 0.1 m2)
    (assert-array-equal m m2))))

(define-test (test-cblas-saxpy)
  (loop-subtests (tn)
  (let* ((rows (logand #xfffe (inexact->exact (truncate (+ 2 (* 512 (random-uniform)))))))
         (a 0.01)
         (rx (gpu-make-vector rows))
         (ry (gpu-make-vector rows)))
    (L 2 "saxpy rows: ~a~%" rows)
    (gpu-array-apply rx (lambda (x) (* 20 (- (random-uniform) 0.5))))
    (gpu-array-apply ry (lambda (x) (* 20 (- (random-uniform) 0.5))))
    (let ((x2 (array-copy (gpu-array rx)))
          (y2 (array-copy (gpu-array ry))))
      (saxpy! a (gpu-array rx) (gpu-array ry))
      (ref-saxpy! a x2 y2)
      (assert-array-equal (gpu-array ry) y2))
    (gpu-free-array rx)
    (gpu-free-array ry))))

(define-test (test-cblas-saxpy-2)
  (let* (; reference arrays, computed by axiomal ref-saxpy
         (gmhw (rand-m! (make-typed-array 'f32 *unspecified* 40 198)))
         (gmyw (rand-m! (make-typed-array 'f32 *unspecified* 2 40)))
         (mhw  (rand-m! (make-typed-array 'f32 *unspecified* 40 198)))
         (vho  (rand-v! (make-typed-array 'f32 *unspecified* 40)))
         (myw  (rand-m! (make-typed-array 'f32 *unspecified* 2 40)))
         (vyo  (rand-m! (make-typed-array 'f32 *unspecified* 2 40)))
         (go   (rand-v! (make-typed-array 'f32 0. 2)))
         (gho           (make-typed-array 'f32 0. 2 40))
         (vxi  (rand-v! (make-typed-array 'f32 *unspecified* 198)))
         ; copy used by guile-cblas
         (gmhw2 (array-copy gmhw))
         (gmyw2 (array-copy gmyw))
         (mhw2  (array-copy mhw))
         (vho2  (array-copy vho))
         (myw2  (array-copy myw))
         (vyo2  (array-copy vyo))
         (go2   (array-copy go))
         (gho2  (array-copy gho))
         (vxi2  (array-copy vxi)))
    ; need higher amplitude on input-values to detect errors above error-epsilon
    (array-map! vxi (lambda (x) (* x 1000)) vxi)
    (array-map! vxi2 (lambda (x) (* x 1000)) vxi2)

    ;-------------------------------------
    ; reference

    (do ((i 0 (+ i 1))) ((= i 2))
      (ref-saxpy! (array-ref go i) vho (array-cell-ref gmyw i))
      (ref-saxpy! (array-ref go i) (array-cell-ref myw i) (array-cell-ref gho i)))

    (do ((k 0 (+ k 1))) ((= k 2))
      (do ((i 0 (+ i 1))) ((= i 40))
        (ref-saxpy! (array-ref (array-cell-ref gho k) i) vxi (array-cell-ref gmhw i))))


    ;----------------------------------
    ; do guile-cblas version and verify (j iter-variable is done by saxpy)
    ;----------------------------------
    ; loop k = (0, 2)               ; each output neuron
    ; loop j = (0, 40)              ; each hidden output
    ;   gmyw(k,j) += vho(j) * go(k)
    ;   gho(j) += go(k) * myw(k,j)
    (do ((k 0 (+ k 1))) ((= k 2))
      (saxpy! (array-ref go2 k) vho2 (array-cell-ref gmyw2 k))
      (saxpy! (array-ref go2 k) (array-cell-ref myw2 k) (array-cell-ref gho2 k)))

    ;--------------------------------------------------------------------
    ; gmhw(i) <- gw(i) + vxi * gho(i)

    ; loop k  2         ; k = each output neuron
    ; loop i 40         ; i = each hidden neuron
    ; loop j 198        ; j = each network-input
    ;   gmhw(i,j) += vxi(j) * gho(k,i)

    (do ((k 0 (+ k 1))) ((= k 2))
    (do ((i 0 (+ i 1))) ((= i 40))
      (saxpy! (array-ref (array-cell-ref gho2 k) i) vxi2 (array-cell-ref gmhw2 i))))

    ;---------------------------------------------------------------------
    (assert-array-equal gmhw gmhw2)
    (assert-array-equal gmyw gmyw2)
    (assert-array-equal mhw  mhw2)
    (assert-array-equal vho  vho2)
    (assert-array-equal myw  myw2)
    (assert-array-equal vyo  vyo2)
    (assert-array-equal go   go2)
    (assert-array-equal gho  gho2)
    (assert-array-equal vxi  vxi2) ; assert input hasn't been touched
    ))


(define-test (test-cblas-saxpy-3)
  (let* ((alpha 0.9)
         ; reference arrays, computed by axiomal ref-saxpy
         (gmhw (rand-m! (make-typed-array 'f32 *unspecified* 40 198)))
         (gmyw (rand-m! (make-typed-array 'f32 *unspecified* 2 40)))
         (mhw  (rand-m! (make-typed-array 'f32 *unspecified* 40 198)))
         (vho  (rand-v! (make-typed-array 'f32 *unspecified* 40)))
         (myw  (rand-m! (make-typed-array 'f32 *unspecified* 2 40)))
         (vyo  (rand-m! (make-typed-array 'f32 *unspecified* 2 40)))
         (tderr   (rand-v! (make-typed-array 'f32 0. 2)))
         (go   (rand-v! (make-typed-array 'f32 0. 2)))
         (gho           (make-typed-array 'f32 0. 2 40))
         (vxi  (rand-v! (make-typed-array 'f32 *unspecified* 198)))
         ; copy used by guile-cblas
         (gmhw2 (array-copy gmhw))
         (gmyw2 (array-copy gmyw))
         (mhw2  (array-copy mhw))
         (vho2  (array-copy vho))
         (myw2  (array-copy myw))
         (vyo2  (array-copy vyo))
         (tderr2  (array-copy tderr))
         (go2   (array-copy go))
         (gho2  (array-copy gho))
         (vxi2  (array-copy vxi)))
    ; need higher amplitude on input-values to detect errors above error-epsilon
    (array-map! tderr (lambda (x) (* x 1000)) vxi)
    (array-map! tderr (lambda (x) (* x 1000)) vxi2)

    ;-------------------------------------
    ; reference
    ;-------------------------------------
     (match (array-dimensions myw)
       ((r c)
        (do ((i 0 (+ i 1))) ((= i r)) ; i = each output neuron
          (let ((tde (array-ref tderr i)))
          (do ((j 0 (+ j 1))) ((= j c)) ; j = each hidden output
            (let ((w (array-ref myw i j))
                  (e (array-ref gmyw i j)))
              (array-set! myw (+ w (* alpha e tde)) i j)))))))

     (match (array-dimensions mhw)
       ((r c)
        (do ((i 0 (+ i 1))) ((= i r)) ; i = each hidden neuron
          (do ((j 0 (+ j 1))) ((= j c)) ; j = each network-input
            (let ((w (array-ref mhw i j))
                  (e (+ (* (array-ref tderr 0) (array-ref gmhw i j))
                        (* (array-ref tderr 1) (array-ref gmhw i j)))))
              (array-set! mhw (+ w (* alpha e)) i j))))))

    ;----------------------------------
    ; do guile-cblas version and verify (j iter-variable is done by saxpy)
    ;----------------------------------
    (match (array-dimensions myw)
      ((r c)
       (do ((i 0 (+ i 1))) ((= i r)) ; i = each output neuron
         (let ((tde (* alpha (array-ref tderr i))))
           (saxpy! tde (array-cell-ref gmyw2 i) (array-cell-ref myw2 i))))))

     (match (array-dimensions mhw)
       ((r c)
        (do ((i 0 (+ i 1))) ((= i r)) ; i = each hidden neuron
          (let ((tde0 (* alpha (array-ref tderr 0)))
                (tde1 (* alpha (array-ref tderr 1))))
            (saxpy! tde0 (array-cell-ref gmhw2 i) (array-cell-ref mhw2 i))
            (saxpy! tde1 (array-cell-ref gmhw2 i) (array-cell-ref mhw2 i))))))

    ;---------------------------------------------------------------------
    (assert-array-equal gmhw gmhw2)
    (assert-array-equal gmyw gmyw2)
    (assert-array-equal mhw  mhw2)
    (assert-array-equal vho  vho2)
    (assert-array-equal myw  myw2)
    (assert-array-equal vyo  vyo2)
    (assert-array-equal go   go2)
    (assert-array-equal gho  gho2)
    (assert-array-equal vxi  vxi2) ; assert input hasn't been touched
    ))

(define-test (test-cblas-blas-sgemv)
  (loop-subtests (tn)
  (let* ((alpha 1.0)
         (beta  1.0)
         (rows (logand #xfffe (inexact->exact (truncate (+ 2 (* 8 (random-uniform)))))))
         (cols (logand #xfffe (inexact->exact (truncate (+ 2 (* 256 (random-uniform)))))))
         (rm (make-typed-array 'f32 *unspecified* rows cols))
         (rx (make-typed-array 'f32 *unspecified* cols))
         (ry (make-typed-array 'f32 0. rows)))
    (do ((i 0 (1+ i))) ((= i rows))
      (array-set! ry (+ 7 i) i)
      (do ((j 0 (1+ j))) ((= j cols))
        (array-set! rx (- (random-uniform) .5) j)
        (array-set! rm (- (random-uniform) .5) i j)))
    (let ((x2 (array-copy rx))
          (y2 (array-copy ry)))
      (sgemv! alpha rm CblasNoTrans rx beta ry)
      (ref-sgemv! alpha rm #f x2 beta y2)
      (let ((err 0.))
        (array-for-each (lambda (a b)
                          (set! err (+ err (abs (- a b)))))
                        ry y2)
        (L 2 "sgemv dims: ~a, ~a err: ~a~%" rows cols err))
      (assert-array-equal ry y2)))))
