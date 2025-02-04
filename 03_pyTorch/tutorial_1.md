## Index for coding Tutorials: Beginner Course
1. Installation
2. Tensor Basics
3. Autograd
4. Backpropagation
5. Gradient Descent With Autograd and Backpropagation
6. Training Pipeline: Model, Loss, and Optimizer
7. Linear Regression
8. Logistic Regression
9. Dataset and DataLoader
10. Dataset Transforms
11. Softmax And Cross Entropy
12. Activation Functions
13. Feed-Forward Neural Net
14. Convolutional Neural Net (CNN)
15. Transfer Learning
16. Tensorboard
17. Save and Load Models

### PyTorch Tutorial 01 - Installation
At this step, it is beleived that you have anaconda installed \
Create new virtual environment:
```
conda create --name pytorch
```   
Activate environment 
```
conda activate pytorch
```
    
Install Pytorch. Check the PyTorch official website for installating the verison
```
pip install --proxy http://proxyout.lanl.gov:8080 torch torchvision torchaudio
```
    
Test pytorch in python
```
python
import torch
x=torch.rand(3)
print(x)
```
    
Check if CUDA is available 
```
torch.cuda.is_available()
```



