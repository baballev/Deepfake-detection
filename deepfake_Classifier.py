from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import copy
from PIL import Image
import glob # UNIX style path expansion
## GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Custom classes
## NN
class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()
        self.convolutional = torchvision.models.resnet34(pretrained=True)
        self.convolutional.fc = nn.Linear(512, 512)
        self.lstm1 = nn.LSTMCell(512, 512) # Output (ht): [512 + 512, 1]
        self.fc1 = nn.Linear(512+512, 2)

    def forward(x):
        x = F.relu(self.convolutional(x))
        x = F.relu(self.lstm1(x))
        x = F.relu(self.fc1(x))

        return x

## Data sampler
class VideoFolderSampler(torch.utils.data.Sampler):
    def __init__(self, end_idx):
        indices = torch.clone(end_idx[:-1]).detach() # Take all indices of end_idx (except the last one because it's ending indices)
        self.indices = indices[torch.randperm(len(indices))] # We then shuffle these starting indices

    def __iter__(self):
        return iter(self.indices.tolist())

    def __len__(self):
        return len(self.indices)

## Dataset
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, end_idx, transform):
        self.image_paths = image_paths
        self.end_idx = end_idx.clone().detach()
        self.length = len(end_idx) - 1
        self.transform = transform
        self.end_idx_reverse_dict = {} # /!\ Ugly code incoming ? /!\
        for i, ind in enumerate(self.end_idx):
            self.end_idx_reverse_dict[ind.item()] = i

    def __getitem__(self, index):
        start = index
        end = self.end_idx[self.end_idx_reverse_dict[index] + 1]

        indices = list(range(start, end))
        images = []
        for i in indices:
            image_path = self.image_paths[i][0] # 0 for the path, 1 for the label (true / fake)
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image).to(device) # apply normalization
            images.append(image)
        x = torch.stack(images).to(device) # The sequence of images with a new dimension for time (that's why we use stack instead of cat)
        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long).to(device) # The label
        return x, y

    def __len__(self):
        return self.length

### Loading datasets
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

## Training
train_path = 'E:/Programmation/Python/PAF 2020/deepfake2/dataset-paf/v2/train/'
train_class_paths = [d.path for d in os.scandir(train_path) if d.is_dir] # ['./true', './fake']

class_image_paths = [] # This is the list which contains the path to every image in every video and which class is corresponding to this images (0 or 1 corresponding to fake or true)
# ToDo: Check which number corresponds to which class
end_idx = [0]  # List of indices which indicates where each element (video) of the set stops.
# EXAMPLE: 1st video frames are class_path_images[0 : end_idx[0]]
#          2nd video frames are class_path_images[end_idx[0], end_idx[1]]
#          [...]

for i, class_path in enumerate(train_class_paths): # 2 folders, one for each class (fake/true)
    for folder in os.scandir(class_path): # Iterate over videos, 1 folder =1 video
        if folder.is_dir:
            paths = sorted(glob.glob(os.path.join(folder.path, '*.png'))) # sorted list of all the images(frames) path within a folder representing a video
            paths = [(p, i) for p in paths] # i = number of the corresponding class, here there's only 2 classes.
            class_image_paths.extend(paths) # inserts every element of paths in class_images_path
            end_idx.extend([len(paths)]) # remember every video length

end_idx = torch.cumsum(torch.tensor(end_idx).to(device), 0).to(device) # end_idx[i] = sum(from j=0 to j=i of end_idx[j])
train_sampler = VideoFolderSampler(end_idx)

trainset = VideoDataset(image_paths=class_image_paths, end_idx=end_idx, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, sampler=train_sampler)

## Validation
valid_path = 'E:/Programmation/Python/PAF 2020/deepfake2/dataset-paf/v2/valid/'
valid_class_paths = [d.path for d in os.scandir(train_path) if d.is_dir] # ['./true', './fake']

class_image_paths2 = []
end_idx2 = [0]
for i, class_path in enumerate(valid_class_paths): # 2 folders, one for each class (fake/true)
    for folder in os.scandir(class_path): # Iterate over videos, 1 folder =1 video
        if folder.is_dir:
            paths = sorted(glob.glob(os.path.join(folder.path, '*.png')))
            paths = [(p, i) for p in paths]
            class_image_paths2.extend(paths)
            end_idx2.extend([len(paths)])

end_idx2 = torch.cumsum(torch.tensor(end_idx2).to(device), 0).to(device) # end_idx[i] = sum(from j=0 to j=i of end_idx[j])
valid_sampler = VideoFolderSampler(end_idx2)

validset = VideoDataset(image_paths=class_image_paths2, end_idx=end_idx2, transform=transform)
validloader = torch.utils.data.DataLoader(validset, batch_size=1, sampler=valid_sampler)

###
rnn = RNN()
rnn.to(device)
print(rnn)








































