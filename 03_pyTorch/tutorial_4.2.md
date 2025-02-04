# Linear Regression
Linear Regression in Python - Machine Learning From Scratch  https://www.youtube.com/watch?v=4swNt7PiamQ

In Regression we want to predict continuous walues. However in classification we want to predict discrete values like 0 or 1.

__Approximation__: 

$$ \hat{y} = wx + b $$

where, $w$ is the slope and $b$ is the intercept or shift on the y axis for 2D case.
We have to come up with an algorithm to find w and b. For that we have to define cost function. In linear regression it is mean squared Error.

__Cost function__:

$$ MSE = J(w,b) = \frac{1}{N} \sum_{i=1}^{n} (y_i - (wx_i + b))^2 $$

We want to minimise this error. To find the minimum we have to fins d the derivative or gradient. So we want to calculate the gradeint w.r.t. $w$ and $b$

$$ J'(m,b) = $$
