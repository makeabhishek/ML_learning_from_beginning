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
**Generate random numbers** 
```
x = torch.rand(2,2)
print(x)
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

**Check the shape, size of Tensor**
```
print(x.shape())
print(x.size())
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
