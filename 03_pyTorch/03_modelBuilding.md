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

### Simple MLP with three FC Layers in PyTorch
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
1. AlexNet: 
2. ZF Net:
3. GoogLeNet (Inception)
4. Microsoft ResNet
5. Depthwise Seperable Convolutions
6. Xception
7. MobileNet


### AlexNet (2012)
- The first architecture developed in 2012 at the begining. First convolutional netork architecture for computer vision. It is very similar to the architecture used by yann lecun in 1998, but it was much deeper. Additionally they were the first to train the model in ImageNet data (more than 1.4 million).
- The network consists of 5 Conv layers, Max-Pooling layers, Dropout layers, and 3 Fully Connected layers.

![image](https://github.com/user-attachments/assets/535d0868-041b-4d20-b63b-bdb5187eb713) \
Alexnet architecture (May look weird becasue there are two differnet "streams." This is becasue the training process was so computationally expensive that they had to split the training onto two GPU's.)

![image](https://github.com/user-attachments/assets/ce8febab-b6ab-4a23-a99a-c99f135c08d0)

<img width="728" alt="image" src="https://github.com/user-attachments/assets/bd431a73-9b41-47d8-a73b-10bcd810c4d4">

### ZF Net (2013)
<img width="647" alt="image" src="https://github.com/user-attachments/assets/0bcfebad-dd25-4b9d-9f39-19c6cc785f4a"> \
ZF Net Architecture

- Very similar architecture to AlexNet, except for a few minor modifications.
- AlexNet trained on 15 million images, while ZF Net trained on only 1.3 million images.
- __Instead of using 11x11 sized filters in the first layer (which is what AlexNet implemented), ZF Net used filters of size 7x7 and a decreased stride value. The reasoning behind this modification is that a smaller filter size in the first conv layer helps retain a lot of original pixel information in the input volume. A filtering of size 11x11 proved to be skipping a lot of relevant information, especially as this is the first conv layer.__
- As the network grows, we also see a rise in the number of filters used.
- It is this paper which introduces the feature visualization, named __DeConvNet__. Today we call feature visualization as explainability.
- VGG also has a similar learnings (salient feature, simple and deep)

### VGG Net (2014)
From Oxford Vision group. \
There are two architectures:
1. VGG16
2. VGG19
Key features: __No z-scoring, no preprocessing__. Still better than AlexNet and ZF Net

<img width="458" alt="image" src="https://github.com/user-attachments/assets/c9ecdd92-f9cd-499c-b675-75f06cfe1560"> \
The 6 different architectures of VGG Net. Configuration D produced the best results.

### ### GoogLeNet (2015)
- Mixture of differnt convlution size.
- Used 9 Inception modules in the whole architecture, with over 100 layers in total! Now that is deep...
- __No use of fully connected layers!__ They use an average pool instead, to go from a 7x7x1024 volume to a 1x1x1024 volume. This saves a huge number of parameters.
- Uses 12x fewer parameters than AlexNet.

<img width="421" alt="image" src="https://github.com/user-attachments/assets/8dada8b4-cb4f-43b9-ab78-c1c30db85221"> \
One full Inception module

<img width="599" alt="image" src="https://github.com/user-attachments/assets/14f7a194-14df-45e5-96f0-29a75f097faa"> \
Green box shows parallel region of GoogLeNet.

### Microsoft ResNet (2015)
- 152 layers...
- Interesting note that after only the first 2 layers, the spatial size gets compressed from an input volume of 224x224 to a 56x56 volume.
- Authors claim that a naïve increase of layers in plain nets result in higher training and test error (Figure 1: He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.)
- The group tried a 1202-layer network, but got a lower test accuracy, presumably due to overfitting.
- It has intersting architecture, where we discuss about 1x1 convolution to 3x3 convolution, and each residual block takes in the feature from previous layer perform some operation then concatenate and send (figure 1). We can have multiple residual block to create a network

<img width="292" alt="image" src="https://github.com/user-attachments/assets/2b326e08-4290-4220-bdfc-a176209f4fb2">  <img width="194" alt="image" src="https://github.com/user-attachments/assets/57bd279f-a137-4582-bd23-835fd4fe1f6b"> \
Optimized version of ResNet connections by [5] to shield computation

#### ResNets
- ResNets (Residual Networks): Introduced by Kaiming He in 2015. [He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.]
- Address __vanishing gradients in deep networks.__
- Use of residual connections (shortcuts) that __skip layers.__
- Easier learning of __identity mappings,__ allowing deeper models.
- Improved performance and training stability. make optimization landscape wider.
- Widely used in image classification and object detection.
- 
<img width="377" alt="image" src="https://github.com/user-attachments/assets/eb0a0477-07b4-4882-8e20-6a8bacf039a0">

#### Variants of ResNets
- Each variant differs in the number of layers, offering a trade-off between model complexity and computational cost.
- __ResNet-18:__ A smaller version with 18 layers, suitable for lighter tasks.
- __ResNet-34__: An intermediate model with 34 layers, balancing complexity and performance.
- __ResNet-50__: Utilizes bottleneck layers, consisting of 50 layers, commonly used in practice.
- __ResNet-101__: A deeper model with 101 layers, offering higher accuracy.
- __ResNet-152__: The deepest version with 152 layers, used for very complex tasks. Gives the best performing model, even than 200 layers.
These standard architectures are providing standard models that can be used for our applications. These models have been trained on millions of images, we can use transfer learning to train our model and fine tune the model for our task.

### Inception-ResNet (V1-V4) (2016)
Improvement to inception (from Google) and ResNet (from Microsoft)

<img width="206" alt="image" src="https://github.com/user-attachments/assets/7f040298-54bf-49f2-96c7-c52476eddd10">

### Xception (2017)
Using depth wise Separable Convolution layer instead of convolution. It reduces the number of parameters and increase performance.

<img width="447" alt="image" src="https://github.com/user-attachments/assets/9c60fde1-159a-429c-a4ce-7b1bcec2f0ee">

<img width="275" alt="image" src="https://github.com/user-attachments/assets/0b206f66-efa8-4f28-8eb2-ff81f1af684e">

### DenseNet (2017-2018)
Use Dense Layers as connections and concatenate.

<img width="451" alt="image" src="https://github.com/user-attachments/assets/9154069d-d797-403b-88fd-cc374094bcd3">

## General Design Principles
1. Avoid Representational Bottlenecks.
2. Increasing the activations per tile in a convolutional network allows for more disentangled features.
3. Spatial aggregation can be done over lower dimensional embeddings without much or any loss in representational power.
4. Balance the width and depth of the network.

# Remarks on networks

## Comparison
<img width="734" alt="image" src="https://github.com/user-attachments/assets/98ef0c5e-03a9-4b6f-8f45-6917a5720e65">

## Double Descent
As we keep increasing number of parameters of the network or width of network, we will see the reduction of error. but if we keep going forward, we use double descent. This help in more generalization.

![image](https://github.com/user-attachments/assets/aabc3c77-f5dc-40f0-be10-589e13f4866e)


# Analogy of architecture with matehmatics
## ResNet: 
Reference: He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. \

Modelling differential equation. The idea here is to represent $f$ by a deep neural network and we will trian this network using modern numerical integration scheme to step this forward (Figure) and some kind of adjoint sensitivity using the auto differentiability of the neural network that we're going to represent $f$ with. 

If I want to model input-output relationship between some state $x$ maybe at timestep $k$ and at timestep $k+1$. A reasonable way to do this is to actually copy the input directly over and only model the difference between the output and the input the so-called **residual.** This is what we're modeling with our neural network $f$. This allowed us to have much deeper networks without having the problem of forgetting the input over you know those many many layers.

To write this down in kind of math what this ends up saying is that the output ($x_{k+1}$) is equal to input ($x_{k}$) plus the neural network (modeling this residual) $f$, which is a function of the input. Recall teh scientific computng math, this is pretty much a __Euler numerical integrator__. This is how I integrate a differential equation $\dot{x} = f(x)$ in a simplest way.

We know that Euler integration especially for large timestep is pretty bad way to integrate. If a residual network is doing Euler Integration through vector field $f$. Then the thing that neural network is learning this residual can also be
thought of as a vector field uh some differential equation $\dot{x}$ equals $f$ that we are essentially time stepping forward with this residual Network.

![image](https://github.com/user-attachments/assets/6dc8099b-e74b-49a1-b5d9-52d27c5bb3f1)

Euler integration is usually a bad idea it's very quick and dirty but it's also highly prone to instability and it has huge errors it's the worst way you could integrate a particle through a vector. A better way to do this is using Neural ODE.

## Neural ODE
Reference: Chen, Ricky TQ, et al. "Neural ordinary differential equations." Advances in neural information processing systems 31 (2018).

Instead of modeling residual ($f$) using an Euler integration time step, maybe we can instead instead of modeling this large time step per forward map from $x_k$ to $x_{k+1}$, which is the Euler step. We can model the differential equation itself i.e., $\dot{x} = f(x)$ which is the actual Vector field and then use a fancier numerical integration scheme to essentially step that state forward and train $f$ based on my training data. 

So in this neural ODE if I am modeling the right hand side of a continuous time differential equation $\frac{d}{dt} x = f(x)$, whereas Euler integration $x_{k+1}=x_k + f(x_k)$ in ResNet is kind of a discrete time approximation of this continuous time differential equation. I can use the neural network to model $f$ directly and if I have this function $f$ we know from classical mathematics, such as, Newton Lebnitz Euler lagrange mathematics. We know how to solve these differential equations. 

![image](https://github.com/user-attachments/assets/f9b087ce-7b80-4289-95d5-8da72a9743cc)

$𝑥_{𝑘+1} = 𝑥_𝑘 + \int_{t_k}^{t_{k+1}} 𝑓(𝑥(\tau)) 𝑑\tau$

Even in the modern computational era we have a lot more experience in solving these kinds of differential equations by stepping them forward numerically than we have experience with machine learning. Numerical integration is way older than
modern machine learning. Therefore, if I have neural ODE representation of my differential equation i.e, if I learned function $f$ with a neural network I can essentially write down exactly $x_{k+1}$ in Euler integration is in terms of  function $f$ in $x_{k+1} = \int_{t_k}^{t_{k+1}} f(x(\tau)) d\tau$, which is not an approximation. It is the exact expression for $x$ at timestep $k + 1$ given this differential equation $\frac{d}{dt} x = f(x)$. However, this integral operator equation is hard to compute. This is what a mathematician would write down as the solution. But there are better numerical integrators to approximate $x_{k+1}$ (Figure) than that of forward Euler solution.

For example, a second order Runge Kutta scheme, a fourth order Runge Kutta scheme, a simplec integrator or a variational integrator. The good thing about neural OdEs is that we can actually use different integrators and we can get different variance of this algorithm. 

### Summary: 
Neural ODE is inspired by a residual networks. Recognizing that ResNets are essentially fitting data, where data need to be sampled uniformly in time with a fixed timestep between the points and ResNet is about the worst numerical integrator you could think of to try and model any kind of differential equation (Euler integration). So ResNet might not be the best way to model a differential equation.

In contrast, Neural ODE models the actual differential equation on the right hand side itself instead of just a onetime step update or flow map of that differential equation and then once we represent the system this way and try to train thr neural network to represent the continuous RHS (i.e, $f(x)$) of $\frac{d}{dt} x = f(x)$, now we can use different numerical integration schemes to train the parameters of $f$ to be consistent with training data points. 

![image](https://github.com/user-attachments/assets/1e5fdbfe-b5cf-48ce-86b3-dd9a655e5060)

In other words, with neural ODE, I actually can have irregularly spaced points in time. I don't need them to be evenly spaced like in the ResNets because I'm using data points to train the parameters of this neural network $f$ and what I'm doing is essentially tweaking the parameters of $f$ so that when I discretize the numerical integrator (above block diagram in figure) and trying to fir the training data. They fit as accurately as possible so that if I numerically  integrate (using fancy integrator, i.e, ML) along that Vector field I get as close to the data I'm sampling as possible.

We can use all the standard tricks of __Auto differentiation__ and __back propagation__ to tweak those Network parameters for $f$ based on this observational (training) data but under the hood we are time stepping our Vector field $f$ using a fancy better numerical integrator than something like ResNet's Euler Integrator.

![image](https://github.com/user-attachments/assets/6073ab84-1bf4-4f3c-82f8-b32cec412893)

To keep track of hidden state we introduce Lagrange multiplier variable that satisfy the differential equation and we can compute that using adjoint lagrange equation.

![image](https://github.com/user-attachments/assets/b013699d-6ba3-4d5e-a3ba-7863391a929e)

### Comparison between ResNet and NeuralODE
<img width="422" alt="image" src="https://github.com/user-attachments/assets/b7b876a6-d883-4f70-b4b7-635b9bfc9f9c"> \
From Reference paper. 

## Autoencoders
Autoencoders are shown in\
1. D.E. Rumelhart, G.E. Hinton, and R.J. Williams, "Learning internal representations by error propagation." , Parallel Distributed Processing. Vol 1: Foundations. MIT Press, Cambridge, MA, 1986.
2. Ballard, "Modular learning in neural networks," Proceedings AAAI (1987).
3. Bank, Dor, Noam Koenigstein, and Raja Giryes. "Autoencoders." Machine learning for data science handbook: data mining and knowledge discovery handbook (2023): 353-374.

We can think SVD as a very simple autoencoder NN. SVD is also know as PCA/POD.

![image](https://github.com/user-attachments/assets/13a92c7b-8d87-4bdf-8647-c70612891c1d) \
Autoencoder (Shallow, linear)

We generalize this linear coordinate embedding by making it now a deep neural network so instead of one latent space layer now we're going to have many many hidden layers for the encoder many hidden layers for the decoder and our activation units (nodes) are going to have non-linear activation functions. So this allows us to do is instead of learning a linear subspace where our data is efficiently represented now we're able to learn a non-linear manifold  
parameterized by these coordinate $z$ where our data  is efficiently represented and often that means we can get a massive reduction in degrees of freedom  needed to describe this latent space z good and

![image](https://github.com/user-attachments/assets/4061bb2d-5611-4324-a097-a22b47abd282)

latent space can be principal components. 
![image](https://github.com/user-attachments/assets/5b98ae8a-3b89-4e77-b051-1f945dea19f9) \
![image](https://github.com/user-attachments/assets/c5cd273d-0c37-40f9-a0ee-550d0b63c4b8)




