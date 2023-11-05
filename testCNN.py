import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters
num_epochs = 5
batch_size = 10
learning_rate = 0.001

# def get_data():
data_dir = '/Users/chap/BrainTumorDetectionModel/Data'

transform = transforms.Compose(
[transforms.ToTensor(),
 transforms.Resize((256,256)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 

train_set = datasets.ImageFolder(data_dir + '/train_set', transform=transform)
test_set = datasets.ImageFolder(data_dir + '/test_set', transform=transform)

train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # return train, test

# def train_imshow():
classes = (True, False)
dataiter = iter(train)
images, labels = next(dataiter)

conv1 = nn.Conv2d(3,6,5) #in_channels, out_channels, kernel_size
pool = nn.MaxPool2d(6,6) #kernel_size, stride (shift x pixel to the right)
conv2 = nn.Conv2d(6, 16, 5) 
# images = images.numpy()
print(images.shape)
x = conv1(images)
print(x.shape)
x = pool(x)
print(x.shape)
x = conv2(x)
print(x.shape)
x = pool(x)
print(x.shape)

# fig, axes = plt.subplots(figsize=(10, 4), ncols=5)
# for i in range(5):
#     ax = axes[i]
#     ax.imshow(images[i].permute(1,2,0)) #permute to change the order of the axes
#     ax.title.set_text(' '.join('%5s' % classes[labels[i]]))
# plt.show()

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(3,6,5) #in_channels, out_channels, kernel_size
#         self.pool = nn.MaxPool2d(10,10) #kernel_size, stride (shift x pixel to the right)
#         self.conv2 = nn.Conv2d(6, 16, 5) 
#         self.fc1 = nn.Linear(16*5*5, 120) # 5x5 is the size of the image after 2 conv layers, 16 is the number of channels
#         self.fc2 = nn.Linear(120,84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x))) 
#         x = self.pool(F.relu(self.conv2(x))) 
#         x = x.view(-1, 16*5*5)  #flatten
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x






#Data set is split into training and test sets (85% and 15%)
