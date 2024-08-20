This is the repository to get used to PyTorch

### PyTorch Installation

Install anaconda:  https://docs.conda.io/projects/conda/en/23.1.x/user-guide/install/macos.html \
Create conda environment:  ```conda create -n learnpytorch``` \
Activate environment: ```conda activate learnpytorch``` \
Install pip for the environment: ```conda install pip``` \
Install PyTorch in the environment from pytorch website (https://pytorch.org/get-started/locally/): ```pip3 install torch torchvision torchaudio```

check installed environments `conda env list`

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
- Tensors can run on GPU’s unlike numPy array. \

### Tensor Initialization
Before we write the code in PyTorch, lets look how can we initialise the tensors. Here I show four different ways to initilise tensor. Tensor can have any dimension i.e, 1D, 2D, nD. We can initilise the tensor in cpu or gpu, we can initlise based on data type, for example `float`, `long`
- From Numpy: convert using `torch.Tensor` 
- Using torch.Tensor(). we can create a list or list of list or dictionary of list.
- Using random
- Using zeros_like and Ones_like

- Tensors are a core PyTorch data type similar to a multidimensional array (like in Numpy). For example the image shown below can be represented in RGB channel with width and height. The tensore representaion of this image can be written as [3, 224, 224].
- Use for representing data in numerical way
- Tensors can run on GPUs unlike Numpy's array

<img alt="image" src="https://github.com/user-attachments/assets/6c1c1668-a8d7-4b75-8a03-974f6f971a1b" width=70% height=70%>

**Help in Pytorch**
Always good to chekc the arguments in a function
```
help(torch.randint)
```

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

**Create an empty tensor** \
```
# Create tensor ifrom list
a = torch.Tensor([0.1, 0.2, 0.1, 0.4, 155])
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
torch.randn(3,4) # or
b = torch.randn(size=(128,128))
print(b)

# Check size, max elemt, min element of b
b.size(); print('size of b is:', b)
b.max(); print('Maximum value in Tensor b is:', b)
b.min(); print('Minimum value in Tensor b is:', b)

# define datatyep while generating random number
c  = torch.randn(size=(128,128), dtype=torch.float64)
c  = torch.randn(size=(128,128), dtype=torch.intt64) #  Why giving error: because randn and int cannot go together to follow the ditribution

d =  torch.randint(high=100, size=(128,128), dtype=toch.int64) # high is the maximum value of number in the tensor
print(d)
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

### Tensor manipulation
Once we initilise the tensor we can perform differnt type of manipulations on tensors. It can be in three different ways,
- Element-wise operations: if we have two tensor `a` and `b` we can multiply them `a*b`
- Functional operations: we can use pytorch function like `torch.matmul`, `torch.nn.functional.relu`
- Modular operations: this is a object oriented way of performing same functional operations. For example, if we want to pass an image as a convolutional filter then we can create a module for conv filter. That module can be used to perform filtering, whcih comes an output of that module.

__Why need to worry about tensor manipulation__
The reason to consider a particular tensor manipulation is important because of __computational graph__. \
Often when we talk about deep learning or neural networks, we need to optimize for the __weights__ and __biases__ using __forward__ and __backward propagation__. When we perform forward and backward propagation, we need to get __gradient__. To obtain the gradients we need to store a record of different tensors and different operations that we have performed using directed acyclic graph (DAG).

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

**Element wise  operations** 
```
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
z = torch.matmul(x,y)
torch.exp(x)
```

**Functional operations** 
```
# multiply two vectors and check the size
tensor1 = torch.randn(3)
tensor2 = torch.randn(3)
torch.matmul(tensor1, tensor2).size()
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

c = np.arange(100).reshape(25,4)
d =  torch.from_numpy(c)
print(d.type)
print(d.dtype)

# convert to long datatype i.e., 64 bit integer
e = torch.LongTensor(d)
e.dtype

# convert to float tensor
f = torch.FloatTensor(d+0.1) # adding 0.1 to hav ein the float.
f.dtype

# convert to double tensor i.e., 
g = torch.DoubletTensor(d.astype()Torch.Float64)+0.1) #  Why this error 
g.dtype

g = d.type(torch.float64)
g.dtype
```

## 3. Gradinet Calculation with autograd
Differentiation is a crucial step in nearly all machine learning and deep learning optimization algorithms. While the calculations for taking these derivatives are straightforward, working out the updates by hand can be a tedious task. We will get a conceptual understanding of 
how __autograd__ works to find the __gradient__ of multivariable functions. \
We will discuss some fundamentals on _derivatives_, _partial derivatives_, _gradients_, and _Jacobians_. We then discuss how to compute _gradients_ using ```requires_grad=True``` and the ```backward()``` method. 
Thus, we cover ```classes``` and ```functions``` implementing __automatic differentiation__ of arbitrary scalar-valued and non-scalar-valued functions. We also discuss the _Jacobian matrix_ in PyTorch. 

### Computational Graph
We discussed in above section, how computational graphs are generated in PyTorch.

- NN optimize weights and biases using forward and backward propagation.
- Deep learnign graphs record tensors and operations in directed acyclic graph (DAG). An example of DAG  is shown in the figure below, where `x`, and `y` are input tensors, which goes into multiplication operation, gives `v`, whcih is then passed to `log`, gives `w`. here multiplication is an element-wise operation, then `log` is a functional oeration. \
In tensorflow these graphs are compiled apriori to start of DL or NN training, whereas,
- PyTorch uses dynamic graphs, built at runtime for efficiency. So because of dynamic graph code actually becomes the graph.
- In the graph, nodes represent operations (functions) and edges represents the tensors. $x,y,v,w$ are nodes in the computational graph. multiplication and log are the operators.

<img alt="image" src="https://github.com/user-attachments/assets/cffe0ab3-0374-4968-aec7-d93a348ceef1" width=50% height=50%>

- Computational graphs are useful in creating `Autograd`. The way it is done that once we build the graph dynamically during tensor operation and recorded all the dependencies. We can now use this dynamic computation graph to compute the gradients for the tensors by travelling in the graph in backward direction. This is performed using __Chain Rule__. Once we compute the backward propagation using chain rule, we get the gradients and once we have have the gradients, the optimization of parameters during model training can go.

### Graph Visualization
- Different ways to visualize the model
- Simple way is to call `print(model)`
        - Quick summary but lacks the pictorial visual. Cant tell whats the input, output of model size, how to interpret model layers. \
        - Not suitable for deep networks 
- Some other options are: \
        - Torchviz: It uses graph based visualization. Gives all the operations performed in PyTorch.  \
        - Tensorboard
- Let’s consider small network with 3 fully-connected layers
 ```
        - input = nn.Linear(in_features=4, out_features=16)
        - hidden_1 = nn.Linear(in_features=16, out_features=16)
        - output = nn.Linear(in_features=16, out_features=3)
 ```

#### Torchviz
__pip install torchviz__

```
from torchviz import make_dot

model = Net()
y=model(X)

make_dot(y.mean(), params=dict(model.named_parameters()))
```
<img width="370" alt="image" src="https://github.com/user-attachments/assets/a0b00375-8236-412c-ae50-84d06ce18f79"> \
Graph Visualization using torch viz for a small network

#### Tensorboard
__pip install tensorboard__
```
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("torchlogs/")
model = Net()
writer.add_graph(model, X)
writer.close()

cd <path-to-logs-dir>
tensorboard --logdir=./
```

<img width="752" alt="image" src="https://github.com/user-attachments/assets/c8eba878-aa1a-4f99-8a3c-9e21c43e8599"> \
Graph Visualization using tensorboard for a small network

### Autograd and Backpropagation
- Automatic differentiation library that facilitates the computation of gradients for tensor operations.
- Dynamic Computational Graph: Builds graph dynamically during tensor operations, recording dependencies. Backward Propagation: Traverses graph backward to compute gradients for tensors
- Gradient Calculation: Computes gradients efficiently, optimizing parameters during model training.
- Ease of Use: Simplifies training loops, automating gradient computation for developers.

**Autograd Example**
Here is an example, how forward anf backward pass works. \
We can compute $\frac{\partial w}{\partial v}$, which is a log derivative. Once we have log derivative, we can compute $\frac{\partial w}{\partial x}$ and $\frac{\partial w}{\partial y}$, whcih is mult derivative, as long as we knwo the way to compute the derivative for multiplication operation.

<img alt="image" src="https://github.com/user-attachments/assets/8d31c817-8b02-47e2-936c-3e7f251f1c67" width=70% height=70%>

In summary,

<img alt="image" src="https://github.com/user-attachments/assets/dfecee58-11cb-447a-9184-73c6cae66264" width=80% height=80%>

## 4. PyTorch Hooks
To debug what happen in forward pass and backward pass. Understand the model internal states. How to keep track of them. 
- PyTorch hooks are functions attached to tensors and modules (layers).
- They allow modification or inspection of outputs and gradients.
- Hooks work during both forward and backward passes.
- They provide a powerful way to interact with the model's internal states.
- Useful for debugging, visualizing, and modifying network behavior during training.

_For example:_ let's we have a linear layer and we want to know the input and output of the linear layer and how is this model performing in between. We can create a hook. If we want only input layer we can create forward hook before the layer is executed, and we can always use another hook after the layer is executed. There is a pre and post hooks and forward and backward hook. These four hook helps us in visualizeing and debugging.

- Two types of hooks in PyTorch
- __Forward Hooks:__ \
        - Forward hooks are triggered during the forward pass. \
        - Activated after the module's forward method is called. \
        - Allow inspection or modification of a layer's output.
- __Backward Hooks:__ \
        - Backward hooks are triggered during the backward pass. \
        - Activated when gradients are being computed. \
        - Allow inspection or modification of gradients for a tensor or layer.


<img width="825" alt="image" src="https://github.com/user-attachments/assets/c2bfeac8-7275-4588-8496-26ada91b1350"> \
The 96 filters learned in the first convolution layer in AlexNet.

## 5. Gradient Descent with Autograd
__Exercise__
_Problem Statement:_  Simple linear regression using gradeint descent.  \
The problem statement is using pytorch tensors to build a simple linear regression model with basic gradient descent. So that we know whats going on under the hood.
- Initializing weights and biases randomly `w` and `b` with `1` and `0`, as intial guess to fit the line `y = 2.25*x + 1.25`, i.e, `y =w*x + c`.
- So from inital guess of `1` and `0` we are moving towards `2.25` and `1.25`.
- Use concepts of Deep learnign to perform this task. Compute the gradients of `w` and `b`
- Define the objective function of compute the loss, `L`. Loss is basically the objective function. Use Means square loss (MSE) ` torch.nn.functional.mse_loss(y_pred,y)`.
- Compute gradeitn of `L` $\frac{\partial L}{\partial w}$, once we compute this, we can do gradient descent,
- $w = w - \alpha \frac{\partial L}{\partial w}$ This is a gradeint descent step.
- Similarly for `b`

Snippet to gradient descent \
![image](https://github.com/user-attachments/assets/04ba7516-51f3-4788-a89c-674dc5e5e27d)


```
import torch

# w = torch.Tensor([1.0], requires_grad = True) # this will give error
w = torch.Tensor([1.0])
w.requires_grad = True
w.type(torch.float64)

b = torch.Tensor([0.0])
b.requires_grad = True
b.type(torch.float64)

# creat some x values
x = torch.Tensor([f*0.0001 for f in range(100000)])
x.type(torch.float64)

# create a line with slope 2.25 and intercept 1.25. 
y = 2.25*x + 1.25

# -------------------------------------------------------#
# Do some prediction 
# y_pred = w*x + b; loss = torch.nn.functional.mse_loss(y_pred,y) # MSE loss between y_pred and y
# loss.backward()
# See some gradeint values
# w.grad
# b.grad
# w = w - 0.01*w.grad # 0.01 is the learning rate
# b = b - 0.01*b.grad
# w
# -------------------------------------------------------#

# Perform above prcedure in for loop
# TASK: change number of iterations and learning rate to see the target value of 2.25 1.25. play with this to reach to them.
num_iter = 500000
for iter in range(num_iter):
        print(iter)
        var_w = torch.autograd.Variable(w,requires_grad=True)
        var_b = torch.autograd.Variable(b,requires_grad=True)
        y_pred = var_w*x + var_b
        loss = torch.nn.functional.mse_loss(y_pred,y)
        loss.backward()
        lr = 3E-4 # 0.0001                                 # 0.01, 0.1 (small learning rate helps to converge for convex optimization.)
        with torch.no_grad(): 
                # var_w = var_w - lr*var_w.grad            # this will give error. change it to inline operation.
                var_w -= lr*var_w.grad                     # as variables are changing dynamically so we have to fine the var_w
                var_b -= lr*var_b.grad
        # Run without making gradients zero and observe the loss value
        var_w.grad.zero_()                                 # make gradients back to zero for next steps. Else it will add values to already existing gradeints.
        var_b.grad.zero_()
        if iter%100==0:
                # if using constant lr as defined in the starting. you can see that the it stuck so uncomment this to see the performance.
                # lr *= 0.9999
                print('Iteration:',iter, 'loss value:',loss.item())

print(var_w, var_b)

```

## GPU usage in PyTorch
Always carefule in transfering data in cpu and GPU. Is it necessary?

```

```


## classes 
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
