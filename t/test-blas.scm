

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
