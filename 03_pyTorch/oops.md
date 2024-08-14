## Object Oriented Programming (OOPs) Concepts:
A class can be thought of as a piece of code that specifies the data and behaviour that represent and model a particular type of object. We can write a fully functional class to model the real world. These classes help to better organize the code and solve complex programming problems.

- Class: 
- Objects
- Data Abstraction 
- Encapsulation
- Inheritance
- Polymorphism
- Dynamic Binding
- Message Passing

(1) A __class__ is a custom data type defined by the user, comprising data members and member functions. These can be accessed and utilized by creating an instance of the class. It encapsulates the properties and methods common to all objects of a specific type, serving as a blueprint for creating objects. __For Example:__ Consider the Class of Cars. There may be many cars with different names and brands but all of them will share some common properties like all of them will have 4 wheels, Speed Limit, Mileage range, etc. So here, the Car is the class, and wheels, speed limits, and mileage are their properties.

(2) An __object__ is an instance of a class. In OOP, an object is a fundamental unit that represents real-world entities. When a class is defined, it doesn't allocate memory, but memory is allocated when it is instantiated, meaning when an object is created. An object has an identity, state, and behaviour, and it contains both data and methods to manipulate that data. Objects can interact without needing to know the internal details of each other’s data or methods; it is enough to understand the type of messages they accept and the responses they return.

For example, a 'Dog' is a real-world object with characteristics such as colour, breed, and behaviours like barking, sleeping, and eating.

(3) The term __methods__ refers to the different behaviours that objects will show. Methods are __functions__ that you define within a class. These functions typically operate on or with the attributes of the underlying instance or class. Attributes and methods are collectively referred to as members of a class or object.

(4) The term __attributes__ refers to the properties or data associated with a specific object of a given class. In Python, attributes are __variables__ defined inside a class to store all the required data for the class to work.

In short, __attributes__ are variables, while methods are __functions__.

__Inheritance:__ Inheritance is an important pillar of OOP. The capability of a class to derive properties and characteristics from another class is called Inheritance. When we write a class, we inherit properties from other classes. So when we create a class, we do not need to write all the properties and functions again and again, as these can be inherited from another class that possesses it. Inheritance allows the user to reuse the code whenever possible and reduce its redundancy.



## Neural Network Example
Creating, training, and testing a neural network in PyTorch involves several key steps. Here's a high-level overview of the process: \
(1) Install PyTorch \
(2) Import Libraries \
(3) Define the Neural Network \
(4) Prepare the Data \
(5) Initialize the Model, Loss Function, and Optimizer \
(6) Train the Model \
(7) Test the Model \
(8) Save and Load the Model (Optional) \

Let's take a look at step 3. Create a class that defines your neural network. This involves inheriting from ```nn.Module``` and implementing the ```__init__``` and ```forward methods```. 

```
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(784, 128)  # Example for MNIST dataset (28x28 images)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)   # 10 classes for MNIST

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

```
Let's break down the concepts of classes, including the ```__init__``` method, the ```forward``` method, and the ```super()``` function, which are essential for creating neural networks in PyTorch.

### Classes in Python
A class in Python is a blueprint for creating objects (instances). Classes encapsulate data and functions that operate on that data, allowing for modular and reusable code. In the context of PyTorch, we use classes to define neural network architectures.

### Key Components
1. `__init__` Method \
The `__init__` method is a special method in Python classes that acts as a constructor. It is automatically called when a new instance of the class is created. This method is used to initialize the attributes of the class.

In the context of a PyTorch neural network:

__Purpose:__ Initialize the layers of the neural network. \
__Usage:__ Define the layers as class attributes.

Example:
```
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()  # Call the parent class's constructor
        self.layer1 = nn.Linear(784, 128)  # Initialize the first layer
        self.layer2 = nn.Linear(128, 10)   # Initialize the second layer
```
2. `forward Method` \
The `forward method` defines the forward pass of the neural network. It specifies how the input data should flow through the network's layers to produce an output.

__Purpose:__ Define the computation performed at every call. \
__Usage:__ Apply layers and activation functions to the input data.

Example:
```
# Input to forward: `x` is passed as an argument to the forward method. It represents the input data being fed into the neural network. x is transfom=rmed by each layer in the network. The final `x` after processing through the network is the output of the forward method, representing the network's predictions for the given input.

def forward(self, x):
    x = self.layer1(x)  # Pass input through the first layer
    x = self.relu(x)    # Apply ReLU activation function
    x = self.layer2(x)  # Pass through the second layer
    return x
```

3. `super()` Function \
The `super()` function is used to call methods from a parent or superclass. This is particularly useful in object-oriented programming to ensure that the parent class's methods are called and initialized properly.

In the context of PyTorch:

__Purpose:__ Call the constructor of the parent class (nn.Module). \
__Usage:__ Ensure that the PyTorch nn.Module is initialized correctly.

Example:
```
def __init__(self):
    super(SimpleNN, self).__init__()  # Initialize the parent class
```
### Summary
`__init__`: Initializes the network's layers. \
`forward`: Defines the data flow through the network. \
`super()`: Calls the parent class's methods to ensure proper initialization. \




