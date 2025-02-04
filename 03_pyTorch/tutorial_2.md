# PyTorch Tutorial 02 - Tensor Basics
#### In pytorch everything is tensor. 
You may be aware of arrays from numpy. Tensor can be of any diemnsion
```
import torch
```
#### Create an empty 1D Tensor
```
x= torch.empty(1)
print(x)
```

#### Create an empty 2D Tensor
```
x= torch.empty(2,3)
print(x)
```
#### Create an empty 3D Tensor
```
x= torch.empty(2,3,3)
print(x)
```

#### Create a random Tensor
```
x= torch.rand(2,2)
print(x)
```

#### Create a zero Tensor
```
x= torch.zeros(2,2)
print(x)
```

#### Create a one Tensor
```
x= torch.ones(2,2)
print(x)
```

#### Give spsecific data type. Check the data typr
```
print(x.dtype) #  by defauult its a float32 data type
```

#### Assign data type to tensor
```
x= torch.ones(2,2, dtype=torch.int)
x= torch.ones(2,2, dtype=torch.double)
x= torch.ones(2,2, dtype=torch.float16)
print(x)
```

#### Check the size
```
print(x.size())
```

##### Create tensor from data, eg. from list
```
x= torch.tensor([2.5, 0.1])
print(x)
```

#### Perform tensor operations
```
x = torch.rand(2,2)
y = torch.rand(2,2)
print(x)
print(y)
z = x + y # element wise addition
z = torch.add(x,y)
print(z)

z = x - y # element wise substracion
z = torch.sub(x,y)
print(z)

z = x * y # element wise multiplication
z = torch.mul(x,y)
print(z)

z = x / y # element wise division
z = torch.div(x,y)
print(z)
```

#### inplace addition: this will modify y by adding all the elements of x to y
#### In pytorch everyfunction whcih has trailing underscore (_) do a inplace operation
```
y.add_(x) 
print(y)
y.sub_(x) 
print(y)
y.mul_(x)
print(y)
```

#### Slicing in pytorch
```
x = torch.rand(5,3)
print(x)
print(x[:, 0]) # all the rows but first column only
print(x[0, :]) # 1st row all the column
print(x[1, 1]) # only ine element
# if tensor has only one element we can also call.item method, whcih give the actual value. Only used if on value in tensor
print(x[1, 1].item()) # only ine element
```

#### Reshaping tensors: number of elements must be same
```
x = torch.rand(4,4)
y = x.view(16)
print(y)
# if we don't want to put number of element for diemenison we can also use
y = x.view(-1,8)
print(y)
print(y.size())
```

#### Converting numpy to torch
```
import numpy as np
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
print(type(b))
```

#### We have to be careful, because if the tensor is on CPU not on GPU than both objects share the same memory location. If we change one we ahve to change other.
```
a.add_(1)
print(a)
print(b) # we cna see it will also add 1 to b.
```

#### Conver numpy to torch: If we have numpy array in the begineeing
```
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

a +=1
print(a) 
print(b) # tensor is also odified automaitcally.

if torch.cuda.is_available():
    device = toech.device("cuda")
    x = torch.ones(5, device=device) # create a tensor and put it in GPU
    y = torch.ones(5)
    # move to device GPU
    y = y.to(device)
    # now do operation, whcih will be perform in GPU
    z = x + y
    # z.numpy() # this will retrun an error because numpy can only handle CPU tensors. So you cannot convert GPU bacjk to numpy
    # We have to move back to CPU
    z = z.to("cpu")
    z.numpy() 
```

#### when we need to calculate the gradient, later for the optimization step.
```
x = torch.ones(5, requires_grad=True)
```

print(x)
