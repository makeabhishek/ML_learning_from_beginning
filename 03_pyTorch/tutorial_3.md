# PyTorch Tutorial 03 - Gradient Calculation With Autograd
Lets see the autograd package in pytorch and see how we can calculate gradeints from it. Gradients are important for model optimization

```
import torch
x = torch.randn(3)
print(x)

# If we need to Calcualte gradients of some fucntion w.r.t. x, we give additonal arguemnts to tensor
x = torch.randn(3, requires_grad=True)
print(x)
```

whenever we will do a computaiton with this rtensor. Pytor ch will create a computaional graph.
let's do some operation

```
y = x + 2 
```

We do an operation y = x+ 2
It will create a computaitonal graph. For each operation we have a nodes with input and output. Here operation is additon 
We have input x and 2; output is y. Pytorch creae a computaitonal graphy using use backpropagation to calcaulte the gradients
--------------------------------------------------------------------------
|                           y = x + 2                                    |              
|                ----------> Forward -------->                           |
|               (x)                                                      |
|    I             -                                             O       |
|    N               -                                           U       |
|    P                -> (+) --> y --> ------------              T       |
|    U              -                             |              P       |
|    T            -                               | grad_fn      U       |
|    S         (2)                                |              T       |
|               <------ Add Backward (dy/dx) <-----                      |
-------------------------------------------------------------------------- 
First we do a Forward Pass. We calcaute the output y. Sisnce we specified that it 
requires the gradient. Pytorch will automatically creates and store a fcuntion for us.
This funsiton is than used in the backppropagation to get the gradeints. Here `y` has an attribute `grad_fn`
So this will point to a gradient function. Her it is called `Add Backward`, with this fucntion we can than 
calculate the gradeints, so called backward path. this will calcualte gradeint `y` w.r.t. `x` i.e., dy/dx

In the background it basically creates a vector Jacobian products to get the gradients, whcih will look like
We have Jacobain matrix of partial derivatives, whcih we multiply it with gradient vector we will get the final gradeitns whcih we are intersted in. Also called Chain rule

        \frac{\partial y_1}{\partial x_1}  . . . \frac{\partial y_m}{\partial x_m} 
                        .               .                       .
J.v =                   .                   .                   .                   (\frac{\partial l}{\partial y_1 ... \frac{\partial l}{\partial y_m}})^T = (\frac{\partial l}{\partial x_1 ... \frac{\partial l}{\partial x_n}})^T 
                        .                       .               .
        \frac{\partial y_1}{\partial x_n}  . . . \frac{\partial y_m}{\partial x_n}

So note that we have to multiply `J` with vector `v`
However in above case z is a scalar value so we dont need to use argument for our backward funciton. But for vector lets see

"""

```
print(x)
# >>>tensor([ 0.1585,  0.1784, -1.3230], requires_grad=True)
print(y) #  Gadient fucntion is AddBackward
#>>>tensor([2.1585, 2.1784, 0.6770], grad_fn=<AddBackward0>)

z = y*y*2
print(z) # Gadient fucntion is MulBackward0
#>>>tensor([9.3183, 9.4910, 0.9166], grad_fn=<MulBackward0>)

# if z is scalar
z = z.mean()
print(z) # Gadient fucntion is MeanBackward0
#>>>tensor(6.5753, grad_fn=<MeanBackward0>)

# to calculate gradients, we jsut need to do
z.backward() # Calcualte graditnet of z w.r.t. x dz/dx
print(x.grad)

# if z is vector. In the backgraound it a jacobian vector product so we have to give vecctor
z = y*y*2
z.nackward() # it will give error. So we have to give vector of same size
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v) #dz/dx
```

#### Prevent pytorch from tracking history and calculating `grad_fn` attribute. 
For example, sometimes in our trainign loop. when we update the weights, this operaion should not be part of gradeint computaiton. We can do it in three ways
Stop pytorch in creat grad_fn
```
# x.requires_grad_(False)
# x.detach()
# with torch.no_grad():

x = torch.randn(3, requires_grad=True)
print(x)
    
x.requires_grad_(False) # rememebr whenever we have fucntion with underscore at the end it will modify our varaible inplace.
print(x)

y = x.detach()
print(y)

with torch.no_grad():
    y = x + 2
    print(y)
```

whenever we call the backward fucntion than the grdient for the tensor is accumulated in the .grad attribut. So the values must be summed up
#### lets create soem dummy training example
```
weights = torch.ones(4, requires_grad=true)

for epoch in range(1):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    
for epoch in range(2):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
```
We can see all the values are summed up and gradeints are not correct. So before we do the next iteration in optimization 
step we must empty the gradeitn
```
for epoch in range(2):
    model_output = (weights*3).sum()
    model_output.backward()
    
    print(weights.grad)
    
    weights.grad.zero_()
    
    print(weights.grad)

# later we will work on pytorch builtin optimizer. So we have to do the same.
weights = torch.ones(4, requires_grad=true)
optimizer = toch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()
```
