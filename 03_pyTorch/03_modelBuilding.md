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

# Applying PyTorch for Computer Vision Applications
We will look the real life applications.

## Computer Vision
 - Computer vision is the art of teaching a computer to see.
 - Mimic human vision to automate tasks like image classification, object detection, and scene understanding. 

<img width="262" alt="image" src="https://github.com/user-attachments/assets/d0d4b848-02cd-4813-872c-7394f9310352"> \
Source: computer vision tutorials

## Applications of Computer Vision
- Healthcare: medical imaging, automatic tumour detection
- Agriculture: disese detection
- Insurance: 
- Manufacturing: prediction of tool life, defect detection, assmbly
- Banking: stock market
- Automotive
- Sports: tracking of ball
- Survilance: face detection

## Architectures
1. AlexNet: The first architecture developed in 2012 at the begining. First convolutional netork architecture for computer vision. It is very similar to the architecture used by yann lecun in 1998, but it was much deeper. Additionally they were the first to train the model in ImageNet data (more than 1.4 million).
2. ZF Net:
3. GoogLeNet (Inception)
4. Microsoft ResNet
5. Depthwise Seperable Convolutions
6. Xception
7. MobileNet


### AlexNet (2012)
The network consists of 5 Conv layers, Max-Pooling layers, Dropout layers, and 3 Fully Connected layers.

![image](https://github.com/user-attachments/assets/535d0868-041b-4d20-b63b-bdb5187eb713) \
Alexnet architecture (May look weird becasue there are two differnet "streams." This is becasue the training process was so computationally expensive that they had to split the training onto two GPU's.)

<img width="728" alt="image" src="https://github.com/user-attachments/assets/bd431a73-9b41-47d8-a73b-10bcd810c4d4">

### ZF Net (2013)
<img width="647" alt="image" src="https://github.com/user-attachments/assets/0bcfebad-dd25-4b9d-9f39-19c6cc785f4a"> \
ZF Net Architecture

- Very similar architecture to AlexNet, except for a few minor modifications.
- AlexNet trained on 15 million images, while ZF Net trained on only 1.3 million images.
- Instead of using 11x11 sized filters in the first layer (which is what AlexNet implemented), ZF Net used filters of size 7x7 and a decreased stride value. The reasoning behind this modification is that a smaller filter size in the first conv layer helps retain a lot of original pixel information in the input volume. A filtering of size 11x11 proved to be skipping a lot of relevant information, especially as this is the first conv layer.
- As the network grows, we also see a rise in the number of filters used.
- It is this paper which introduces the feature visualization, named DeConvNet
- VGG also has a similar learnings (salient feature, simple and deep)

### VGG Net (2014)
There are two architectures:
1. VGG16
2. VGG19
Key features: No z-scoring, no preprocessing

<img width="458" alt="image" src="https://github.com/user-attachments/assets/c9ecdd92-f9cd-499c-b675-75f06cfe1560"> \
The 6 differetn architectures of VGG Net. Configuration D produced the best results.

### ### GoogLeNet (2015)
- Used 9 Inception modules in the whole architecture, with over 100 layers in total! Now that is deep...
- No use of fully connected layers! They use an average pool instead, to go from a 7x7x1024 volume to a 1x1x1024 volume. This saves a huge number of parameters.
- Uses 12x fewer parameters than AlexNet.

<img width="421" alt="image" src="https://github.com/user-attachments/assets/8dada8b4-cb4f-43b9-ab78-c1c30db85221"> \
Full Inception module

<img width="599" alt="image" src="https://github.com/user-attachments/assets/14f7a194-14df-45e5-96f0-29a75f097faa"> \
Green box shows parallel region of GoogLeNet.

### Microsoft ResNet (2015)

#### ResNets
- ResNets (Residual Networks): Introduced by Kaiming He in 2015. [He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.]
- Address vanishing gradients in deep networks.
- Use of residual connections (shortcuts) that skip layers.
- Easier learning of identity mappings, allowing deeper models.
- Improved performance and training stability.
- Widely used in image classification and object detection.
- 
<img width="377" alt="image" src="https://github.com/user-attachments/assets/eb0a0477-07b4-4882-8e20-6a8bacf039a0">

#### Variants of ResNets
- Each variant differs in the number of layers, offering a trade-off between model complexity and computational cost.
- __ResNet-18:__ A smaller version with 18 layers, suitable for lighter tasks.
- __ResNet-34__: An intermediate model with 34 layers, balancing complexity and performance.
- __ResNet-50__: Utilizes bottleneck layers, consisting of 50 layers, commonly used in practice.
- __ResNet-101__: A deeper model with 101 layers, offering higher accuracy.
- __ResNet-152__: The deepest version with 152 layers, used for very complex tasks.

### Inception-ResNet (V1-V4) (2016)
Improvement to inception and ResNet

<img width="206" alt="image" src="https://github.com/user-attachments/assets/7f040298-54bf-49f2-96c7-c52476eddd10">

### Xception (2017)
Separable Convolutions

<img width="447" alt="image" src="https://github.com/user-attachments/assets/9c60fde1-159a-429c-a4ce-7b1bcec2f0ee">

<img width="275" alt="image" src="https://github.com/user-attachments/assets/0b206f66-efa8-4f28-8eb2-ff81f1af684e">

### DenseNet (2017-2018)
Use Dense Layers as connections and concatenate

<img width="451" alt="image" src="https://github.com/user-attachments/assets/9154069d-d797-403b-88fd-cc374094bcd3">

## General Design Principles
1. Avoid Representational Bottlenecks
2. Increasing the activations per tile in a convolutional network allows for more disentangled features
3. Spatial aggregation can be done over lower dimensional embeddings without much or any loss in representational power.
4. Balance the width and depth of the network.

# Remarks on networks

## Comparison
<img width="734" alt="image" src="https://github.com/user-attachments/assets/98ef0c5e-03a9-4b6f-8f45-6917a5720e65">

## Double Descent
![image](https://github.com/user-attachments/assets/aabc3c77-f5dc-40f0-be10-589e13f4866e)



