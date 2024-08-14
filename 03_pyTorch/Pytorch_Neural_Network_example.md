Let's walk through a fully connected NN \
__Notes:__ \
Inheritance allows us to define a class that inherits all the methods and properties from another class. The parent class is the class being inherited from, also called a base class. The child class is the class that inherits from another
class, also called the derived class. 

__Import necessary packages__
```
import torch.nn as nn    #
import torch.optim as optim                  # all optimization algorithms
import torch.nn.fucntional as F              # all functions without any parameters
import torch.utils.data import dataloader    # Import data loader to manage data
import torchvision.datasets as dataset       # load datasets available in torch vision
import torchvision.transforms as transforms  # transformations which can be performed on datasets.
```

__Create a Fully connected network (FCN) layers__
```
# Create a class:
class NN(nn.module):   # Create a class to implement FCN. Inherit from nn.module
  def __init__(self, input_size, num_classes): # we have 28x28=784 nodes from MNIST images
    super(NN, self).__init__() # Call super, essentially super calls the initialization method of the parent class, which is nn.module
    # Create a neural network
    self.fc1 = nn.linear(input_size, 50) # hidden layer of 50 nodes
    self.fc2 = nn.linear(50, num_classes)

  def forward(self,x):   # perform forward on some input x, then we want to perform the layers that we initialised in the init method
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# For testing: Lets test that on random data to check if it is giving correct values
#model = NN(784, 10)   # 10 for number of digit
#x = torch.randn(64, 784)
#print(model(x).shape)
    
```
__Set Devices__
```
device = torch.device('cuda' if torch.cuda.is_avaialble() else 'cpu')
```
__Hyper parameters__
```
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1
```
__Load Data__
```
train_dataset = datasets.MNIST(root='dataset/', train=True, transform-transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle = True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform-transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size= batch_size, shuffle = True)
```
__Initiate network__
```
model == NN(input_size = input_size, num_classes=num_classes).to(device)
```
__Loss and optimizer__
```
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate)
```
__Train the neural network__
```
for epoch in range(num_epochs):
  for batch_idx, (data, targets) in enumerate(train_loader):
    data =data.to(device=device)
    targets = targets.to(device=device)
    # Change the shape
    data = data.reshape(data.shape([0],-1))
    print(data.shape)

    # Forward
    scores = model(data)
    loss = criterion(scores, targets)

    # backward
    optimizer.zero_grad()  # set all the gradients to zero for all the batch and does not store from previous
    loss.backward()

    # Gradeitn descent or adam step
    optimizer.step()
  
```
# Check accuracy on trining and test to see how good our model
```
def check_accuracy(loader, model):
  if loader.dataset.train:
    print("Checking accuracy on training data")
  else:
    print("Checking accuracy on testing data")
  num_correct = 0
  num_samples = 0
  model.eval()

  with torch.no_grad():
    for x, y in loader:
      x = x.to(device=device)
      y = y.to(device=device)
      x = x.reshape(x.shape[0],-1)

      scores = model(x)
      _, predictions = scores.max(1)
      num_correct += (predictions == y).sum()
      num_samples += predictions.size(0)

  print(f'Got {num_correct}/ {num_samples} withaccuracy {floadt(num_correct)/float(num_samples)*100:.2f})

  model.train()
```
