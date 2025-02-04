# PyTorch Tutorial 04_a - Backpropagation - Theory With Example


    x----> [a(x)] ----> y ----> [b(y)] ----> z 
    
#### CHAIN RULE:
Lets say we ahve two operations. W ehave input x and apply a fucntion a(x) whcih give output y, 
whcih is further pass to fucntion b(y) and give fuinal output z. We want to minimize z w.r.t. x. dz/dx =?
We can do this using Chain rule
dz/dx = dz/dy.dy/dx

#### COMPUTATIONAL Graph (CG):
    Next thing is CG, so for every operations we do with our tensors, pytorch will create a CG. For each node we apply some operation and we get an output.
    Here we did multiply operation of x and y so the node has multiply operator.
    --------------------------------------------------------------------------
    |                           y = x + 2                                    |              
    |                ----------> Forward -------->                           |
    |               (x)                                                      |
    |    I             -  \frac{\partial z}{\partial x} = \frac{\partial x.y}{\partial x}  = Y        
    |    N               -                                                   |
    |    P                -> (*)f=x.y --> y Loss \frac{\partial loss}{\partial z}
    |    U              -                          
    |    T            -  \frac{\partial z}{\partial y} = \frac{\partial x.y}{\partial y}  =X   
    |    S         (2)                                                    |
    |               <------ Add Backward (dy/dx) <-----                      |
    --------------------------------------------------------------------------
For these nodes we can calculate so called local gradients. and we can use them in the chain rule to get final gradient
We have to calculate lgradeint of loss w.r.t. our paramter in the begineing i.e, input
\frac{\partial loss}{\partial x} =\frac{\partial loss}{\partial z}.\frac{\partial z}{\partial x}

The whole concept consist of three steps
(1) Forward Pass: Compute Loss
(2) Compute Local Gradients
(3) Backward Pass: Compute dLoss/dWeight using chain rule


# Example: Linear Regression
Linear Regression in Python - Machine Learning From Scratch 02 - Python Tutorial https://www.youtube.com/watch?v=4swNt7PiamQ
In Regression we want to predict continuous walues. However in classification we want to predict discrete values like 0 or 1.
Approximation : \hat{y} = wx + b, w is the slope and b is the intercept or shift on the y axis for 2D case.
We have to come up with an algorithm to find w and b. For that we have to define cost function. In linear regression it is mean squared Error.
Cost function: 
    MSE = J(w,b) = \frac{1}{N} \sum_{i=1}^{n} (y_i - (wx_i + b))^2
    We want to minimise this error. To find the minimum we have to fins d the derivative or gradient. So we want to calculate the gradeint w.r.t. w and b
    J'(m,b) = 



# We model our output with a linear combination of some weights and input so \hat{y} = w.x. We formualte the loss ficntion. Lets assume is squared error. 
# Loss = predicted y - actual y)^2
# Loss = (\hat{y} - y)^2
# To minimise the loss we apply three steps.

```
import torch
```
x = torch.tensor(1.0)
y = torch.tensor(2.0)
```

# This is the parameter we want to optimize -> requires_grad=True
w = torch.tensor(1.0, requires_grad=True)

# forward pass to compute loss
y_predicted = w * x
loss = (y_predicted - y)**2
print(loss)

# backward pass to compute gradient dLoss/dw
loss.backward()
print(w.grad)

# update weights
# next forward and backward pass...

# continue optimizing:
# update weights, this operation should not be part of the computational graph
with torch.no_grad():
    w -= 0.01 * w.grad
# don't forget to zero the gradients
w.grad.zero_()

# next forward and backward pass...

```

