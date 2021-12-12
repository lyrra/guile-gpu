(define-module (guile-gpu t test-common)
  #:use-module (ice-9 match)
  #:use-module (guile-gpu common)
  #:use-module (guile-gpu common-lisp)
  #:use-module (guile-gpu gpu)
  #:export (*test-verbose* *test-depth* *current-test* *test-totrun* *test-totrun-subtest*
            *tests*
            test-env? test-env-set
            test-assert-arrays-equal assert-array-equal
            test-assert
            define-test
            loop-subtests
            run-tests
            L))

(define *test-verbose* 1) ; increased verbosity
(define *test-depth* 25)

(define *tests* '())
(define *current-test* #f)
(define *test-totrun* 0)
(define *test-totrun-subtest* 0)
(define *test-env* '())

(define (test-env? key)
  (assoc-ref *test-env* key))

(define (test-env-set key val)
  (set! *test-env*
        (if (assoc key *test-env*)
            (assoc-set! *test-env* key val)
            (acons key val *test-env*)))
  #f)

(define* (assert-array-equal arra arrb #:optional (eps 0.003))
  (array-for-each (lambda (a b)
                    (assert (> eps (abs (- a b)))
                            (format #f "[~f /= ~f] (epsilon: ~f)" a b eps)))
                  arra arrb))

(define-syntax L
  (lambda (x)
    (syntax-case x ()
      ((_ 2 e e* ...) #'(if (and *test-verbose* (>= *test-verbose* 2)) (format #t e e* ...)))
      ((_ e e* ...) #'(if *test-verbose* (format #t e e* ...)))
      ((_ e ...) #'(if *test-verbose* (format #t e ...))))))

(define-syntax define-test
  (lambda (x)
    (syntax-case x ()
      ((_ (proc) e ...)
       #'(begin
           (define (proc)
             (set! *current-test* (procedure-name proc))
             (L "-- running test ~a~%" *current-test*)
             (set! *test-totrun* (1+ *test-totrun*))
             e ...)
           (register-test 'proc proc)
           #f)))))

(define-syntax loop-subtests
  (syntax-rules ()
      ((_ (i) . body)
       (begin
         (do ((i 0 (1+ i)))
             ((>= i *test-depth*))
           (set! *test-totrun-subtest* (1+ *test-totrun-subtest*))
           . body)
         (L "  -- test ~a completed ~a subtests~%"
            *current-test* *test-depth*)))))

(define (register-test name test)
  (set! *tests* (acons name test *tests*)))

(define (test-assert exp . reason)
  (if (not (eq? exp #t))
    (begin
      (format #t "Test ~a has failed.~%" *current-test*)
      (cond
       ((procedure? (car reason))
        ((car reason)))
       (else
        (apply format (append (list #t) reason))))
      (newline)
      (exit))))

(define (epsilon? a b eps)
  (> eps (abs (- a b))))

(define (test-assert-arrays-equal arrs brrs epsilon)
  ; ensure equal amounts of arrays
  (test-assert (= (length arrs) (length brrs)))
  ; ensure each array is equal dimension
  (for-each (lambda (arr brr)
              (test-assert (equal? (array-dimensions arr)
                                   (array-dimensions brr))
                           "not-equal-dimensions"))
            arrs brrs)
  ; ensure array content is equal
  (for-each (lambda (arr brr)
              ;(format #t "arr: ~s~%" arr) (format #t "brr: ~s~%" brr)
              (match (array-dimensions arr)
                ((r c)
                 (do ((i 0 (1+ i))) ((= i r))
                 (do ((j 0 (1+ j))) ((= j c))
                   (let ((a (array-ref arr i j))
                         (b (array-ref brr i j)))
                     (test-assert (epsilon? a b epsilon)
                                  (format #f "i:~a,j:~a[~f,~f]" i j a b))))))
                ((r)
                 (do ((i 0 (1+ i))) ((= i r))
                   (let ((a (array-ref arr i))
                         (b (array-ref brr i)))
                     (test-assert (epsilon? a b epsilon)
                                  (format #f "i:~a[~f,~f]" i a b)))))))
            arrs brrs))

(define (run-test proc)
  (let ((start (gettimeofday))
        (stop #f))
    (proc)
    (set! stop (gettimeofday))
    (format #t "    time: ~s~%"
            (- (+ (* (car  stop) 1000000) (cdr stop))
               (+ (* (car start) 1000000) (cdr start))))))

(define (run-tests)
  (loop-for proc in *tests* do
    (L "run test: ~s~%" (car proc))
    (run-test (cdr proc)))
  (L "tot: ~a, subtests: ~a~%" *test-totrun* *test-totrun-subtest*)
  #t)
