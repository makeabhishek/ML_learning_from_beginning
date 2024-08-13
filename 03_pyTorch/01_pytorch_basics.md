This is the repository to get used to PyTorch

### PyTorch Installation

Install anaconda:  https://docs.conda.io/projects/conda/en/23.1.x/user-guide/install/macos.html \
Create conda environment:  ```conda create -n learnpytorch``` \
Activate environment: ```conda activate learnpytorch``` \
Install pip for the environment: ```conda install pip``` \
Install PyTorch in the environment from pytorch website (https://pytorch.org/get-started/locally/): ```pip3 install torch torchvision torchaudio```

**Launch python shell to test pyTorch** \
```python``` \
**Import pytorch** \
```import torch``` \
```torch.__version__``` \
'2.3.1' \
```torch.cuda.is_available()``` \
False 

Check all the packages installed in the evirenmont: pip3 list

## 1. Overview
Pytorch was developed by Facebook AI. But now owned by pytorch foundation (like Linux)

We should consider to learn
1.	Basics of PyTorch tensors and computational graphs
2.	Implement deep learning models with custom data loaders and preprocessing techniques using PyTorch.
3.	Understand GPU and CPU utilization and PyTorch software infrastructure

### Why to use PyTorch? It's advantages:
1.	Easy to use GPU computational power without going deep into hardware and memory.
2.	Automatic differentiation (AD): Compiled computational graph. We use AD, which is a modular way of computing automatic differentiation. We do it either by compiling computational graph or we use dynamical computational graph. PyTorch uses dynamical computational graph is faster and dynamically changing the graph instead of using single graph for everything.
3.	PyTorch Ecosystem: Tochvision, torchtext, torchaudio.Parallel computing, model deployement.


## 2. Tensor Basics
•	Tensors can run on GPU’s unlike numPy array. \
__Tensor initialization:__ Tensor can have any dimension i.e, 1D, 2D, nD \
•	From numpy: convert using ```torch.Tensor()``` \

**Create an empty tensor** \
Notice the size of tensors in the output
```
#  Create an empty tensor of size 1, so it is a scalar. Do not initialise the value. \
>>> x = torch.empty(1)
>>> print(x)
tensor([0.])
#  Create an empty tensor of size 3, so it is a 1D vector.
>>> x = torch.empty(3)
>>> print(x)
tensor([0., 0., 0.])
#  Create an empty tensor of size 2,3, so it is a 2D matrix.
>>> x = torch.empty(2, 3)
>>> print(x)
tensor([[0., 0., 0.],
        [0., 0., 0.]])
# similarly for 3D
>>> x = torch.empty(2,2, 3)
>>> print(x)
tensor([[[0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.]]])
```
**Genrate numbers or tensors**
```
x= torch.arange(12)
print(x)
# Check the type of tensor
print(type(x))
# Check the dimension of tensor
print(x.shape)         # we dont use () because its a property not a function
# Check the number off elements in 
x.numel()        # total number of elements in one axis
```
**Generate random numbers** 
```
x = torch.rand(2,2)
print(x)
```
**Generate random numbers with noraml distribution, which foloow standard Gaussian Distribution with mean 0 and STD 1** 
```
torch.randn(3,4)
```
**Using zeros and Ones**
```
>>> x = torch.zeros(2,2)
>>> print(x)
tensor([[0., 0.],
        [0., 0.]])

>>> x = torch.ones(2,2)
>>> print(x)
tensor([[1., 1.],
        [1., 1.]])

# Multidimensional tensor. Lets define 3 dimentional tensor. 
y= torch.zeros((2,3,4))
print(y)         # check the paranthesis
# access the elements
y[0]        # extract elemtn from 1st axis

z = torch.ones((2,3,4))
print(z)
```

**Check tensor data type and modify it**
```
>>> x = torch.ones(2,2)
>>> print(x.dtype)
torch.float32  # this is default
>>> x = torch.ones(2,2, dtype=torch.int)
>>>  print(x.dtype)
>>> x = torch.ones(2,2, dtype=torch.double)
>>>  print(x.dtype)
>>> x = torch.ones(2,2, dtype=torch.float16)
>>>  print(x.dtype)
```
**Reshape Tensor**
```
X = x.reshape(3,4) # number of elements must be same
# We can see after reshaping that we have extra [], because of an extra axis
print(X.shape)         # first axis has 3 elements and 2nd axis has 4 elements. 0th axis is the column (y direction)
```

**Check the shape, size of Tensor**
```
print(x.shape())
print(x.size())
```

**Creating Tensor from Python List** 
```
torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
```
**Element wise operations** 
```
x= torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x+y, x-y, x*y, x/y,x**y
torch.exp(x)
```

**Concatenate tensor** 
```
X = torch.arange(12, dtype=torch.flat32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3],[1, 2, 3, 4],[4, 3, 2, 1]])
X,Y
X.shape, Y.shape

torch.cat((X,Y),dim=0) # Concatenate in y direction or column wise
torch.cat((X,Y),dim=1) # Concatenate in x direction or column wise
```
**Other operations**
```
X.sum()         # Summing all the elements in the tensor yields a tensor with only one element
X.sum().shape        # we will not see any number, this mean
X.sum(dim=0)
X.sum(dim=0, keepdim=True)
X.sum(dim=0).shape
X.shum(dim=0, keepdim=True).shape
```

**Perform elementwise operations by invooking the braodcasting mechanism**
```
a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))
a,b
# we cannot do element wise operation as the size is not same. But we can do this using broadcasting. In braodcasting we make the replica of row or column to make the two tensors of same shape.

a+b
```
**Indexing and assignment**
```
X = torch.arange(12, dtype=torch.flat32).reshape((3,4))
X[0]         # the tensor has two axis. If we just index one number , we are indexing based on 0-axis
X[-1]        # index for last set of element
# Assignment
X[1:3] =12
```

**Memory Usage in Pytorch**
```
# Assign the results of an operation to a previously allocated array with slice notation
print('id(X):', id(X))        # id function can be used, whcih give a pointer and memory where tensor is stored
X[:] = X + Y
print('id(X):', id(X))

print('id(X):', id(X))
X += + Y
print('id(X):', id(X))

X[:] = X + Y
X = X + Y
print('id(X):', id(X)) # now we will have different memory
```

**Convert numpy array to torch tensor and viceversa**
```
A =  X.numpy()                # tensor to numpy
B = torch.from_numpy(A)       # tensor from numpy
type(A), type(B)
```
__Tensor Manipulation__ \
•	Element wise operation: every element of $a$ is multiply by $b$ \
•	Functional operations ```torch.mul()``` \
•	Modular operations: object orientated way of performing functional operations \


## 3. Gradinet Calculation with autograd

## 4. Backpropagation 

## 5. Gradient Descent with Autograd










If you define a function inside a class, it’s called a __method__. Methods are used by instances of a class.

```
class Demo:
  def method(self):
    print(‘test’)

demo = Demo()
demo.method()
# this line above is shorthand for this:
Demo.method(self=demo)
```
