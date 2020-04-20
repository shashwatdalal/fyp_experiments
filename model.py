
"""## Part 2

In this part, you will train a ResNet-18 defined on the CIFAR-10 dataset. Code for training and evaluation are provided.

### Your Task

1. Train your network to achieve the best possible test set accuracy after a maximum of 10 epochs of training.

2. You can use techniques such as optimal hyper-parameter searching, data pre-processing

3. If necessary, you can also use another optimizer

4. **Answer the following question:**
Given such a network with a large number of trainable parameters, and a training set of a large number of data, what do you think is the best strategy for hyperparameter searching? 

One strategy for evaluating optimal hyperparameters is using Bayesian optimization. This method involves, placing a prior on the space of hyperparaters and using bayesian inference to update the probabilities on the hyperparameters. The probabilities inferred reflect the likelihood of hyperparameters explaining the data. 

Having a large number of training parameters and a large training set implies training the model till convergence will take a long time. To perform bayesian inference in a feasible time, the inference can be conducted on a model that is trained on a randomly sampled subset of the training data. A distribution can be placed on the training sample so that the probability update on the hyperparameters takes account for the uncertainty of the subset not reflecting the full dataset. Hyperparameters to be tested can be sampled from this evolving probability space. 

This method allows the space of hyperparameters to be randomly sampled in feasible time whilst also outputting the uncertainty of the hyperparameters sampled.
"""

import torch
from torch.nn import Conv2d, MaxPool2d
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

"""Next, we define ResNet-18:"""

# define resnet building blocks

class ResidualBlock(nn.Module): 
    def __init__(self, inchannel, outchannel, stride=1): 
        
        super(ResidualBlock, self).__init__() 
        
        self.left = nn.Sequential(Conv2d(inchannel, outchannel, kernel_size=3, 
                                         stride=stride, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel), 
                                  nn.ReLU(inplace=True), 
                                  Conv2d(outchannel, outchannel, kernel_size=3, 
                                         stride=1, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel)) 
        
        self.shortcut = nn.Sequential() 
        
        if stride != 1 or inchannel != outchannel: 
            
            self.shortcut = nn.Sequential(Conv2d(inchannel, outchannel, 
                                                 kernel_size=1, stride=stride, 
                                                 padding = 0, bias=False), 
                                          nn.BatchNorm2d(outchannel) ) 
            
    def forward(self, x): 
        
        out = self.left(x) 
        
        out += self.shortcut(x) 
        
        out = F.relu(out) 
        
        return out


    
    # define resnet

class ResNet(nn.Module):
    
    def __init__(self, ResidualBlock, num_classes = 10):
        
        super(ResNet, self).__init__()
        
        self.inchannel = 64
        self.conv1 = nn.Sequential(Conv2d(3, 64, kernel_size = 3, stride = 1,
                                            padding = 1, bias = False), 
                                  nn.BatchNorm2d(64), 
                                  nn.ReLU())
        
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride = 1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride = 2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride = 2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride = 2)
        self.maxpool = MaxPool2d(4)
        self.fc = nn.Linear(512, num_classes)
        
    
    def make_layer(self, block, channels, num_blocks, stride):
        
        strides = [stride] + [1] * (num_blocks - 1)
        
        layers = []
        
        for stride in strides:
            
            layers.append(block(self.inchannel, channels, stride))
            
            self.inchannel = channels
            
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.maxpool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        
        return x
    
    
def ResNet18():
    return ResNet(ResidualBlock)

"""### Loading dataset
We will import images from the [torchvision.datasets](https://pytorch.org/docs/stable/torchvision/datasets.html) library <br>
First, we need to define the alterations (transforms) we want to perform to our images - given that transformations are applied when importing the data. <br>
Define the following transforms using the torchvision.datasets library -- you can read the transforms documentation [here](https://pytorch.org/docs/stable/torchvision/transforms.html): <br>
1. Convert images to tensor
2. Normalize mean and std of images with values:mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
"""

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data import ConcatDataset

import torchvision.datasets as dset

import numpy as np

import torchvision.transforms as T

##############################################################
#                       YOUR CODE HERE                       #       
##############################################################

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]


transformation_aug = T.Compose([
    T.RandomResizedCrop(28),
    T.Resize(32),
    T.RandomRotation(10),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])

transformation = T.Compose({
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
})

##############################################################
#                       END OF YOUR CODE                     #
##############################################################

"""Now load the dataset using the 

*   List item
*   List item

transform you defined above, with batch_size = 64<br>
You can check the documentation [here](https://pytorch.org/docs/stable/torchvision/datasets.html)
Then create data loaders (using DataLoader from torch.utils.data) for the training and test set
"""

##############################################################
#                       YOUR CODE HERE                       #       
##############################################################

data_dir = './data'

data_set_train = dset.CIFAR10(root=data_dir, transform=transformation, train=True, download=True)
data_set_train_aug = dset.CIFAR10(root=data_dir, transform=transformation_aug, train=True, download=True)
data_set_test = dset.CIFAR10(root=data_dir, transform=transformation, train=False, download=True)

loader_train = DataLoader(dataset=ConcatDataset([data_set_train,data_set_train_aug]), batch_size=64, shuffle=True)
loader_test = DataLoader(dataset=data_set_test, batch_size=64, shuffle=True)


##############################################################
#                       END OF YOUR CODE                     #       
##############################################################

USE_GPU = True
dtype = torch.float32 

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
    

print_every = 100
def check_accuracy(loader, model):
    # function for test accuracy on validation and test set
    
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

        

def train_part(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API. 
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    writer = SummaryWriter()
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    train_losses = []
    test_losses = []
    for e in range(epochs):
        train_loss = 0
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            loss.backward()

            # Update the parameters of the model using the gradients
            optimizer.step()
            train_loss += loss.item()
            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                writer.add_scalar('Loss/train', loss.item())
                #check_accuracy(loader_val, model)
        
        test_loss = 0 
        for x,y in loader_test:
          x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
          y = y.to(device=device, dtype=torch.long)

          scores = model(x)
          loss = F.cross_entropy(scores, y)
          test_loss += loss.item()
          writer.add_scalar('Loss/test', loss.item())


        train_losses.append(train_loss / len(loader_train))
        test_losses.append(test_loss / len(loader_test))
    return train_losses, test_losses

# code for optimising your network performance

##############################################################
#                       YOUR CODE HERE                       #       
##############################################################



##############################################################
#                       END OF YOUR CODE                     #
##############################################################


# define and train the network
model = ResNet18()
optimizer = optim.Adam(model.parameters())

train_losses, test_losses = train_part(model, optimizer, epochs = 10)


# report test set accuracy

check_accuracy(loader_test, model)


# save the model4
torch.save(model.state_dict(), 'model.pt')
