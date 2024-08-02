estimate <- function(n) {
  return (4 * sum((runif(n)^2 + runif(n)^2) < 1)/n)
}

pi_estimate1 = estimate(10)
pi_estimate2 = estimate(100000)
pi_estimate3 = estimate(10000000)
print(pi_estimate1)
print(pi_estimate2)
print(pi_estimate3)


## bonus problem:

estimate_elipse_area <- function(n, a, b){
  x <- runif(n, 0, a)
  y <- runif(n, 0, b)
  return (4*(sum((x^2/a^2 + y^2/b^2) <= 1)/n)*a*b)
}

ellipse_estimate = estimate_elipse_area(10, 2, 1)
print(ellipse_estimate)
ellipse_estimate = estimate_elipse_area(100000, 2, 1)
print(ellipse_estimate)
ellipse_estimate = estimate_elipse_area(10000000, 2, 1)
print(ellipse_estimate)

