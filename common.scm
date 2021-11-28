(define-module (guile-gpu common)
  #:use-module (ice-9 match)
  #:export (assert
            init-rand
            random-uniform))

(define *verbose* #f)

(define* (assert expr #:optional errmsg)
  (if (not expr)
      (begin
        (if (and errmsg *verbose*)
          (format #t "  error: ~s~%" errmsg))
        (error "Fatal error."))))


(define *randstate* #f)

(define* (init-rand #:optional seed)
  (set! *randstate*
        (seed->random-state (or seed (current-time))))
  #f)

(define (random-uniform)
  (random:uniform *randstate*))

(define (random-number len)
  (random len *randstate*))

(define* (assert expr #:optional errmsg)
  (if (not expr)
      (begin
        (if errmsg
          (format #t "  error: ~s~%" errmsg))
        (error "Fatal error."))))

(define-syntax LLL
  (lambda (x)
    (syntax-case x ()
      ((_ e ...)
       #'(if *verbose* (format #t e ...))))))

