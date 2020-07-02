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

        self.lstm1 = nn.LSTM(512, 600, 3, batch_first=True) # with batch_first=True, data is of shape: (batch_size, seq_len, features)
        # If data is of shape (seq_len, batch_size, features) then batch_first = False

        self.fc1 = nn.Linear(600, 2)

    def forward(self, inputs):
        batch_size, seq_length, channel_size, height, width = inputs.size()
        c_in = inputs.view(batch_size*seq_length, channel_size, height, width) # Remove the temporal dimension within the video with the view and batch_size * video_length on the first dim
        c_out = self.convolutional(c_in)
        x = c_out.view(batch_size, seq_length, -1) # Get the vector ready to go throught the LSTM
        out, (h_n, h_c) = self.lstm1(x)
        # Optionally, the 2nd arg is (h_0, c_0), the initial hidden state for each element in the batch and the intial cell state for each element in the batch
        # (h_n, c_n) represents the value of the hidden state and the cell state at the last (seg_length'th) iteration.
        y = self.fc1(out[:, -1, :]) # Only get the last frame output
        return y

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
    def __init__(self, image_paths, end_idx, transform, frame_every_x_frames=2, number_of_frames=150):
        self.period = frame_every_x_frames
        self.number_of_frames = number_of_frames
        self.image_paths = image_paths
        self.end_idx = end_idx.clone().detach()
        self.length = len(end_idx) - 1
        self.transform = transform
        self.end_idx_reverse_dict = {}
        for i, ind in enumerate(self.end_idx):
            self.end_idx_reverse_dict[ind.item()] = i

    def __getitem__(self, index):
        start = index
        end = self.end_idx[self.end_idx_reverse_dict[index] + 1]

        indices = list(range(start, end))
        images = []
        for i in indices:
            if i - start < self.number_of_frames*self.period and i%self.period == 0: # Only keep the first number_of_frames' frames in case the video is long
                image_path = self.image_paths[i][0] # 0 for the path, 1 for the label (true / fake)s
                image = Image.open(image_path)
                if self.transform:
                    image = self.transform(image) # apply normalization
                images.append(image)
        x = torch.stack(images) # The sequence of images with a new dimension for time (that's why we use stack instead of cat)
        y = torch.tensor(self.image_paths[start][1])
        return x, y

    def __len__(self):
        return self.length
## Evaluation
rnn = RNN()
rnn.to(device)
load_path = './weights/ResNet34&3LSTM-6epochs.pth'
rnn.load_state_dict(torch.load(load_path))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
batch_size=4

eval_path = './dataset-paf/v2/eval/'
eval_class_paths = [d.path for d in os.scandir(eval_path) if d.is_dir] # ['./true', './fake']

class_image_paths = []
end_idx = [0]

for i, class_path in enumerate(eval_class_paths):
    for folder in os.scandir(class_path):
        if folder.is_dir:
            paths = sorted(glob.glob(os.path.join(folder.path, '*.png')))
            paths = [(p, i) for p in paths]
            class_image_paths.extend(paths)
            end_idx.extend([len(paths)])

end_idx = torch.cumsum(torch.tensor(end_idx), 0)
eval_sampler = VideoFolderSampler(end_idx)

evalset = VideoDataset(image_paths=class_image_paths, end_idx=end_idx, transform=transform, frame_every_x_frames=2, number_of_frames=20)
evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size, sampler=eval_sampler)

print("Found " + str(len(evalset)) + " videos in " + eval_path)


with torch.no_grad():
    running_corrects = 0.0
    eval_size = len(evalset)
    for i, data in enumerate(evalloader):
        input, label = data[0].to(device), data[1].to(device)
        output = rnn(input)
        _, predicted = torch.max(output, 1)
        running_corrects += torch.sum(predicted == label.data).item()
    eval_acc = running_corrects / eval_size
    print('Evaluation Acc: {:.5f}'.format(eval_acc))



