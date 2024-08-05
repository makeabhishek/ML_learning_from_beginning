## Neural Network
We will create a basic Neural Network (NN) Model and understand some terminologies.

__What is a Neural network:__ A neural network is a machine learning method that employs interconnected _nodes_, or _neurons_, arranged in _layers_ to enable computers to process data similarly to the human brain. This approach, known as _deep learning_, is a subset of artificial intelligence. The figure shows an NN schematic with three input neurons, and two hidden layers, which are fully connected i.e., each neuron is connected with another neuron of the next layer, forward path, and output layer with one neuron.

<img width="764" alt="image" src="https://github.com/user-attachments/assets/85c1bc50-7b6b-469c-a3e1-e59b9e17e1ad">

Creating, training, and testing a neural network in PyTorch involves several key steps. Here's a high-level overview of the process:
(1) Install PyTorch \
(2) Import Libraries \
(3) Define the Neural Network \
(4) Prepare the Data \
(5) Initialize the Model, Loss Function, and Optimizer \
(6) Train the Model \
(7) Test the Model \
(8) Save and Load the Model (Optional) \

### Example: 
Predict Iris Species. Use multilayer perceptrons (Neural Network) to predict the species of the Iris dataset. \

__Task:__ Write a code to create a NN, which takes the __input__, and __Forward__ it to __hidden layer__, then to next __hidden layer__ and finally gives __output__

__Dataset:__ IRIS dataset (https://archive.ics.uci.edu/dataset/53/iris)

The dataset contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are not linearly separable from each other.

### (1) Setup
Make sure PyTorch is installed.

### (2) Import necessary packages or Libraries
```
import torch
import torch.nn as nn
import torch.nn.functional as F     # all functions without any parameters, help in moving forward

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
```

#### (2.1) Set Devices
```
device = torch.device('cuda' if torch.cuda.is_avaialble() else 'cpu')
```

### (3) Define the Neural Network: 
Create Fully connected network (FCN) layers. _Create a model class that inherits `nn.Module`_. Create a class that defines the neural network. This involves inheriting from _nn.Module_ and implementing the __init__ and __forward__ methods.
```
class Model(nn.Module):                      # create a class that inherit nn.module
  # Input layer (4 features of the flower) -->
  # Hidden Layer1 (number of neurons) -->
  # H2 (n) -->
  # output (3 classes of iris flowers)

  def __init__(self, in_features=4, h1=8, h2=9, output_features=3):
    # Initialize the layers of the neural network: initialising the model and passing input arguments. We consider 8 neurons in h1 and 9 neurons       in h2 layer, out has 3 neurons as we have three output features,
    super().__init__() # instantiate our nn.Module
    self.fc1 = nn.Linear(in_features, h1)    # fully connected layer 1, start from input features 'in_features' to hidden layer h1
    self.fc2 = nn.Linear(h1, h2)             # fully connected layer 2, start from hidden layer h1 and move to hidden layer h2
    self.out = nn.Linear(h2, out_features)   # moving from hidden layer h2 to output

  # Now we have a basic model setup. We need a function that moves everything forward. Define the computation performed at every call. Apply layers and activation functions to the input data.
  def forward(self, x):
    x = F.relu(self.fc1(x))    # start with layer 1
    x = F.relu(self.fc2(x))    # move to layer 2
    x = self.out(x)            # push to output layer

    return x    
```

### (4). Prepare the Data: 
__Load Data__
```
url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
my_df = pd.read_csv(url)
```
__Check Data and post process__
```
my_df.tail()

# Encoding: Change the last column from strings to integers. 
my_df['variety'] = my_df['variety'].replace('Setosa', 0.0)
my_df['variety'] = my_df['variety'].replace('Versicolor', 1.0)
my_df['variety'] = my_df['variety'].replace('Virginica', 2.0)
my_df
```

__Train Test Split!  Set X, y__
```
X = my_df.drop('variety', axis=1)    # features, all columns except the last one (i.e., variety).
y = my_df['variety']                 # outcome


# Convert these to NumPy arrays
X = X.values
y = y.values

# Train Test Split: test size = 20%, 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
```
     
__Convert X features to float tensors and Convert y labels to tensors long__
```
# We are converting the data type to match it with the input data type provided in the IRIS dataset.
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
```

### (5) Initialize the Model, Loss Function, and Optimizer
__Pick a manual seed for randomization__
Manual seeding to keep random numbers the same.

```
torch.manual_seed(41)

# Create an instance of a model. The model needs to be instantiated so it can be used for both training and inference.
model = Model()    # Turn ON all the Model
```

__Set the criterion of the model to measure the error, how far off the predictions are from the data__ \

`nn.CrossEntropyLoss`: Used for multi-class classification problems. It combines `nn.LogSoftmax()` and `nn.NLLLoss()` in one single class. \
Choose Adam Optimizer, lr = learning rate (if an error doesn't go down after a bunch of iterations (epochs), lower our learning rate). The optimizer is responsible for updating the model's parameters based on the gradients computed during backpropagation. Optimizers use the computed gradients to adjust the weights of the network to minimize the loss

```
criterion = nn.CrossEntropyLoss()      

optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # An adaptive learning rate optimizer that often performs well across a wide range of problems.

print(model.parameters)
```

__Why Initialize model, loss and optimizer before training?__ \
__Model Preparation:__ The model must be instantiated before training so that it can be used to make predictions and compute gradients.

__Loss Calculation:__ The loss function must be defined to measure the model's performance during training, which provides the necessary information for updating the model's weights.

__Parameter Optimization:__ The optimizer must be initialized with the model's parameters so it can adjust them to minimize the loss function.

These components form the core of the __training loop__, where the model's weights are iteratively updated to improve its predictions. Initializing them at this point sets the stage for the training process, allowing you to iteratively optimize the model's performance.

### (6) Train the Model (Neural network)
```
# Epochs? (one run through all the training data in our network)
epochs = 100
losses = []
for i in range(epochs):
  # Go forward and get a prediction
  y_pred = model.forward(X_train) # Get predicted results

  # Measure the loss/error, gonna be high at first
  loss = criterion(y_pred, y_train) # predicted values vs the y_train

  # Keep Track of our losses
  losses.append(loss.detach().numpy())

  # print every 10 epoch
  if i % 10 == 0:
    print(f'Epoch: {i} and loss: {loss}')

  # Do some backpropagation: take the error rate of forward propagation and feed it back through the network to fine-tune the weights.
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
```

__Plot the loss!__
```
plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel('Epoch')
```

### (7) Test the Model
__Evaluate Model on Test Data Set (validate the model on the test set)__
```
# Around 20% of data
with torch.no_grad():                # Basically turn off backpropagation. We don't need in testing
  y_eval = model.forward(X_test)     # X_test are features from our test set, y_eval will be predictions
  loss = criterion(y_eval, y_test)   # Find the loss or error
print(loss)
```
__The loss is not close. So we correct__
```
# How to network
correct = 0
with torch.no_grad():
  for i, data in enumerate(X_test):
    y_val = model.forward(data)

    if y_test[i] == 0:
      x = "Setosa"
    elif y_test[i] == 1:
      x = 'Versicolor'
    else:
      x = 'Virginica'

    # Will tell us what type of flower class our network thinks it is
    print(f'{i+1}.)  {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}')

    # Correct or not
    if y_val.argmax().item() == y_test[i]:
      correct +=1

print(f'We got {correct} correct!')
```

```
new_iris = torch.tensor([4.7, 3.2, 1.3, 0.2])
```

```
with torch.no_grad():
  print(model(new_iris))
```

newer_iris = torch.tensor([5.9, 3.0, 5.1, 1.8])

with torch.no_grad():
  print(model(newer_iris))

torch.save(model.state_dict(), 'my_really_awesome_iris_model.pt')
     
__Load the Saved Model__
new_model = Model()
new_model.load_state_dict(torch.load('my_really_awesome_iris_model.pt'))

__Make sure it loaded correctly__
new_model.eval()
     
