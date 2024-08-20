# Pytorch debugging Techniques

- Debugging in PyTorch is essential for:
  - _Identifying Errors:_ Fixing bugs like incorrect tensor shapes, faulty operations, tenosrs in different devices (GPU or CPU).
  - _Ensuring Correctness:_ Verifying model implementation and component functionality.
  - _Optimizing Performance:_ Improving efficiency by identifying bottlenecks. Like improving loops
  - _Understanding Behavior:_ Gaining insights into layer interactions and data flow.
  - _Monitoring Training:_ Ensuring correct learning and making necessary adjustments.
  - _Improving Generalization:_ Addressing overfitting or underfitting issues. What optimizer cause overfitting and underfitting.
  - _Developing Custom Components:_ Ensuring custom layers or functions integrate correctly.
  - _Validating Results:_ Increasing confidence in model predictions and outputs.
- Debugging contributes to robust, efficient, and accurate PyTorch models. 


Computational graph visualization and PyTorch hooks can be used to debug the codes. To understand the input, ouput, layers and operatons happening between layers.

# Complex Neural Network Architectures
Library: `torch.nn` library

- `torch.nn` is the fundamental library and in this library `torch.nn.module` is the fundametal module, that becomes the fundamental class on whihc we build any layers, objects usefule for defining NN architecture.
- 
## torch.nn: A Neural networks Library
Everything inside torch.nn.module is object oriented. The objects that are useful in defining NN are shown below:

<img width="709" alt="image" src="https://github.com/user-attachments/assets/2b0be6e4-3411-4e7d-a6df-1fec099a0f97">

The functional class are available in `torch.nn.module.functional`.

## torch.nn.Module

- __Layer Implementation:__ Can create fully connected, convolutional, pooling layers, and activation functions.
- __Combining Modules:__ Multiple nn.Module objects can be combined to form a larger nn.Module object.
- __Implementing Networks:__ This method allows for building neural networks with many layers.
- The nn.Module class has two methods that you have to override. The structure look like, \
  - `__init__()`:
    - This function is invoked when you create an instance of the nn.Module.
    - define the various _parameters of a layer_ such as filters, kernel size for a convolutional layer, dropout probability for the dropout layer. These all parameters can be initilized here even if we are creating our convolution kernel, that need to be initialized here
  - `forward()`:
    - Define how the output is computed. In the `__init__()` we have deined bunch of layers but how the differnet layer that we defined in `__init__()` are connected with each other is looked in `forward()` function
    - No need for explicitly call this function

### Simple MLP with 3 FC Layers in PyTorch
```
import torch
import torch.nn as nn
import torch.optim as optim

# Define the MLP class
class MLP(nn.Module):
  # lets assume we have some input_size (eg, 32) and i'm transforming it to size of hidden_size1 (eg, 16), that is again trasnformed into hidden_size2 (eg, 8) and 8 dimentisional vector is trnasformed into 4 dimensional vector.
  def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
    super(MLP, self).__init__()
    # Initializing differnet layers, whcih will be used in forward function
    self.layer1 = nn.Linear(input_size, hidden_size1)
    self.layer2 = nn.Linear(input_size, hidden_size2)
    self.layer3 = nn.Linear(input_size, output_size)
    self.relu = nn.ReLU() # I can skip nn.ReLU() and use f.relu 

  def forward(self, x):  
    x = self.layer1(x)  # some input `x` is going in layer 1 and giving you another `x`
    x = self.relu(x)    # the `x` is transformed by reLU and giving another `x`
    x = self.layer2(x)
    x = self.relu(x)
    x = self.layer3(x)
    return x
```

## How to manage PyTorch Layers?
We can create either  `torch.nn.ModuleList` or `torch.nn.Sequential`, whcih will assume that there is a sequentiallity in the forward pass, like we defined in example where we have performd steps in sequentially\
```
def forward(self, x):  
    x = self.layer1(x)  # some input `x` is going in layer 1 and giving you another `x`
    x = self.relu(x)    # the `x` is transformed by reLU and giving another `x`
    x = self.layer2(x)
    x = self.relu(x)
    x = self.layer3(x)
    return x
```
- Using that same idea we can use `torch.nn.Sequential`, whcih will give ius the same results. It essentially chain the operations to automatically construct the forward pass. So using `torch.nn.Sequential` we are avoiding creating a forward pass, rather we are using the automated one. It may not be always usefule especially when we write our custom NN architecture.
- There is also `torch.nn.ModuleList` and `torch.nn.ModuleDict`. `torch.nn.ModuleList` is essentially creating a list of modules. Example, if I have multiple networks we can keep track of those networks by creating Module List.or we can use module dict if we want them in terms of dictionalry

|           | `torch.nn.Sequential`    | `torch.nn.ModuleList` |
| --------  | ----------------         | --------------------  |
| __Purpose:__ | Designed for creating a simple feed-forward neural network by stacking layers sequentially.     | Designed for holding a list of layers or modules, providing flexibility to define complex or dynamic  architectures.   |
| __Usage:__ | Automatically constructs the forward pass by chaining the operations in the order they are added. | Does not define a forward method; you need to manually implement the forward pass and decide how to use the modules. |
| __Initialization:__ | Layers are added in a specific order, and the input flows through each layer sequentially. | Layers are stored in a list, and you can iterate over them or use them selectively in the forward  method.    |

## torch.nn vs torch.nn.functional

|           | `torch.nn`    | `torch.nn.functional` |
| --------  | ----------------         | --------------------  |
| __Purpose:__ | Contains classes that represent layers and modules in neural networks     |  Contains functions that represent operations and activations used in neural networks.   |
| __Usage:__ | Primarily used to define network architectures by creating instances of layer classes.   | - Used for defining operations directly in the forward method without creating layer instances. |
| __Examples:__ |  `nn.Linear, nn.Conv2d`, `nn.ReLU` |  `F.linear, F.conv2d`, `F.relu` |
| __Initialization:__ | Requires creating layer objects and initializing them in the `__init__` method of your custom `nn.Module` class. | Does not require initialization in the `__init__` method. Functions are called directly in the forward method.  |

## Example of Sequential vs ModuleList

| `torch.nn`       | `torch.nn.functional` |
| ---------------- | --------------------  |
|<img width="256" alt="image" src="https://github.com/user-attachments/assets/2710ec13-e2ad-44f0-aac7-fcd74044b532"> | <img width="293" alt="image" src="https://github.com/user-attachments/assets/fc4345ed-7d43-4274-add4-a71a5b21a23c"> | 
|Two-layer network using torch.nn.Sequential | Two-layer network using torch.nn.ModuleList|

