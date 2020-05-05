
(define (test-blas-sscal)
  (let* (; reference arrays, computed by axiomal ref-saxpy
         (v (rand-v! (make-typed-array 'f32 0. 16)))
         (m (rand-m! (make-typed-array 'f32 0. 16 32)))
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
    (assert-array-equal m m2)))

(define (test-blas-saxpy)
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


(define (test-blas-saxpy-2)
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
