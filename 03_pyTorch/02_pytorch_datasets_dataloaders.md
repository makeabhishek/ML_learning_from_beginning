
## PyTorch Datasets and Dataloaders
Earlier we discussed that pytorch tensors can be manipulated Tensor manipulation using Element-wise operations, Functional operations and Modular operations. Functions can be used anywhere, and also modules can be used naywhere. Datasets came under modular appraoch. Data Preprocessing is also a module.

- Datasets
- Dataloaders
- Custom datasets and dataloaders
- Preprocessing

To understand PyTorch, we need to know modeules. How to create modules in PyTorch, as it is a object oriented langauge. Therefore it is important to know about Datasets, Dataloaders, and custom datasets and dataloaders. Another reason to understnad these concepts is it helps in creating own pipeline (DL code) from scratch.

### Datasets and Dataloaders
We could have differnt and large data sets, such as cats and dogs dataset. wheree the images can be of jpeg, png and other format. Also we need to annotate the data sets based on calsee such as cats=0, dogs = 1. So processing data samples can become messy and hard to maintain using manual task. In PyTorch it is not recommended to use such kind of approach. It prefers you create a dataclass class and makes it easy to use. It can be extended to any kind of datasets. Basically, it encapsulating all the information and giving it in one simple API o interface that can work for all the applications.

- Processing data samples can become messy and hard to maintain
- Ideally, dataset code should be decoupled from model training code for better readability and modularity
- PyTorch provides __two data primitives:__ `torch.utils.data.DataLoader` and `torch.utils.data.Dataset`
- `Dataset` calss is where dataset stores the samples and their corresponding labels
- `DataLoader` wraps an iterable around the `Dataset’ to enable easy access to the samples. For example accessing minibatch of samples to perform different task. The reason to use `DataLoader` is that we can use minibatches of data in processing. It aslo has impact on how GPU utilization works and how stochastic gradient descent works.
- These tools allows to use pre-loaded datasets (e.g. FashionMNIST dataset) as well as your own data.

### Standard Datsets
- PyTorch domain libraries provide a number of standard datasets
- Different modalities of standard datasets are available in PyTorch
    - Image dataset
    - Text dataset
    - Audio dataset

<img width="356" alt="image" src="https://github.com/user-attachments/assets/4719f381-5545-4f6d-8762-c0e85197cb64">\
Example of FashionMNIST dataset available in PyTorch
 
### Custom Datset
- A custom Dataset class must implement three functions:
    - __init__() class (it almost used in any class creating in Python)
        - __init__ method in Python is used to initialize objects of a class. It is also called a constructor. Constructors are used to initialize the object’s state. The task of constructors is to initialize(assign values) to the data members of the class when an object of the class is created
        - This function is run once when instantiating the Dataset object
        - We initialize the directory containing the images, the annotations file, and both transforms
        - The method is useful to do any initialization you want to do with your object.
    - __len__()
        - The __len__ function returns the number of samples in our dataset.
    - __getitem__()
        - Given an index it loads and returns a sample from the dataset at the given index
        - It is a double underscore methods, whcih hels in defining array. Just like we access differetn elements in numpy nd array, same way we can acess different elemetns in dataset class by defining __getitem__() method.

The advantage of using datset class instead of loading whole dataset in one go and goinfg forward. When we use dataset class, it can scale for very large datasets like Imagenet (around 14 million images). So we can not load full images in our dataset. We have to load small amount of  data.
Custom dataset aloows to load tht datasets on the fly while using __getitem__() method. So instead of loading all the datasets, we can load the metadata in the __init__() class and using the metadata, whenever an index is send to __getitem__(). It can load the data and retrun it.

__Example__
```
# Define a function to load test and train data.
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Loas MNIST data from 'path'"""
    labels_path = os.path.join(path,
                                '%s-labels-idx1-ubyte.gz'
                                % kind)
    images_path = os.path.join(path,
                                '%s-images-idx3-ubyte.gz'
                                % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.unit8,
                                offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.unit8,
                                offset=16.reshape(len(labels), 784))

    return images, lables
```

```
# Define dataset class
class FMNIST(torch.utilis.data.Datasets):
    def __init__(self,path,kind='train',device='cpu'):
        super().__init__()
        self.images, self.labels = load_mnist(path, kind=kind)
        self.device = device

    def __getitem__(self, idx):
        image, label = slef.images[idx,:], slef.labels[idx]
        image = torch.FloatTensor(image), view(-1,28,28).to(self.device)
        label = torch.from_numpy(np.array([label])).to(self.device)

    def __len__(self):
        return self.images.shape[0]
```

### Dataloaders
Dataloader create different batches of datasets and each of these batches are used for minibatch stochastic gradient descent (below Figure). Once all the batches are iterated it completes one epoch. Then dataloader shuffle the data to create another batches and so on ...
- The `Dataset` retrieves dataset features and labels one sample at a time
- During model training, we pass samples in "minibatches"
- We reshuffle the data at every epoch to reduce overfitting
- Python's _multiprocessing_ is used to speed up data retrieval. While the training is done on one minibatch of data, it tries to retrieve other minibatch of data. It helps to utilize GPU efficiently.
- `DataLoader` is an iterable that abstracts this complexity with an easy API.

<img width="355" alt="image" src="https://github.com/user-attachments/assets/20c13ed2-ab1f-40e8-a81d-cec64654805d">

### Data Pre-processing
Data often needs preprocessing before training machine learning algorithms There are multiple preprocessign steps can be performed on data depending on the need and model architecture. There are no set rules. For example, w can normalize the data.
- Transforms are used to manipulate data for training suitability
- All TorchVision datasets have `transform` and `target_transform` parameters
- `transform` modifies the features, while `target_transform` modifies the labels
- Both parameters accept callables containing the transformation logic
- The `torchvision.transforms` module provides several commonly-used transforms

__Data Augmentation:__ Augmenting the original data from the realsitic data which is also realistic,. Data augmentation assist in making model more robust. Example of dfferent transformation on cat image is shown below. These transforms can be performed in dataset class. \
<img width="511" alt="image" src="https://github.com/user-attachments/assets/79cb4dc6-e6cf-48e6-b35b-05e0f89fe4fd"> <img width="346" alt="image" src="https://github.com/user-attachments/assets/5b6b5e3f-1fb9-4aee-a000-38baf1e37d89">



## Exercise
Load MNIST dataset to train and test model. Create custom network. \
Create a class called `Net` and we are subclassing it with `torch.nn.Module`. `torch.nn.Module` is a base calss whcih provides a lot of features for forward, backward computation and all. It essentially provides module for computing different layers and necessary items for NN.

```

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Instantiating different layers that will be used in network. But I have not provided any pipeline about the structure of the NN, which layer will come first and last.
        # We will do this in forward function or forward path, where we define how the architecture moves.
        self.conv1 = torch.nn.conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4*4*50, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```






