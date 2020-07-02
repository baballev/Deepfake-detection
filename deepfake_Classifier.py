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
"""
Notes on data loading in this script:
    The data set we used is composed of videos represented as folders containing ordered .png images.

This is an example of structure that would work:

    .../dataset-path/train/
            .../dataset-path/train/true/

                    .../dataset-path/train/true/video0/

                            .../dataset-path/train/true/video1/0.png
                            .../dataset-path/train/true/video1/1.png
                            .../dataset-path/train/true/video1/2.png
                            .../dataset-path/train/true/video1/3.png
                                    [...]
                            .../dataset-path/train/true/video1/289.png

                    .../dataset-path/train/true/video1/

                            .../dataset-path/train/true/video1/0.png
                            .../dataset-path/train/true/video1/1.png
                            .../dataset-path/train/true/video1/2.png
                            .../dataset-path/train/true/video1/3.png
                                    [...]
                            .../dataset-path/train/true/video1/171.png

                    .../dataset-path/train/true/video2/

                    .../dataset-path/train/true/video3/
                            [...]
                    .../dataset-path/train/true/video1245/

            .../dataset-path/train/fake/

                    .../dataset-path/train/true/video0/

                            .../dataset-path/train/true/video1/0.png
                            .../dataset-path/train/true/video1/1.png
                            .../dataset-path/train/true/video1/2.png
                            .../dataset-path/train/true/video1/3.png
                                    [...]
                            .../dataset-path/train/true/video1/98.png
                    .../dataset-path/train/true/video1/
"""

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


### Loading datasets
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
batch_size=4

## Training
train_path = './v2/train/'
train_class_paths = [d.path for d in os.scandir(train_path) if d.is_dir] # ['./true', './fake']

class_image_paths = [] # This is the list which contains the path to every image in every video and which class is corresponding to this images (0 or 1 corresponding to fake or true)
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

end_idx = torch.cumsum(torch.tensor(end_idx), 0) # end_idx[i] = sum(from j=0 to j=i of end_idx[j])
train_sampler = VideoFolderSampler(end_idx)

trainset = VideoDataset(image_paths=class_image_paths, end_idx=end_idx, transform=transform, frame_every_x_frames=2, number_of_frames=20)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)

print("Found " + str(len(trainset)) + " videos in " + train_path)

## Validation
valid_path = './dataset-paf/v2/valid/'
valid_class_paths = [d.path for d in os.scandir(valid_path) if d.is_dir] # ['./true', './fake']

class_image_paths2 = []
end_idx2 = [0]
for i, class_path in enumerate(valid_class_paths): # 2 folders, one for each class (fake/true)
    for folder in os.scandir(class_path): # Iterate over videos, 1 folder =1 video
        if folder.is_dir:
            paths = sorted(glob.glob(os.path.join(folder.path, '*.png')))
            paths = [(p, i) for p in paths]
            class_image_paths2.extend(paths)
            end_idx2.extend([len(paths)])
del paths
del valid_class_paths

end_idx2 = torch.cumsum(torch.tensor(end_idx2), 0) # end_idx[i] = sum(from j=0 to j=i of end_idx[j])
valid_sampler = VideoFolderSampler(end_idx2)

validset = VideoDataset(image_paths=class_image_paths2, end_idx=end_idx2, transform=transform, frame_every_x_frames=2, number_of_frames=20)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, sampler=valid_sampler)

print("Found " + str(len(validset)) + " videos in " + valid_path)
del class_image_paths
del class_image_paths2
del end_idx
del end_idx2

### Training
## Train function
def trainModel(model, loss_function, optimizer, epochs_nb):
    since = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_size = len(trainset)
    valid_size = len(validset)
    print("Training start")
    for epoch in range(epochs_nb):
        # Verbose 1
        print("Epoch [" + str(epoch+1) + " / " + str(epochs_nb) + "]")
        print("-" * 10)

        # Training
        running_corrects = 0
        running_loss = 0.0
        verbose_loss = 0.0
        for i, data in enumerate(trainloader):
            input, label = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output, 1) # max dumped in _, argmax in predicted
            print("Batch " + str(i) + " / " + str(int(train_size/batch_size)))
            running_loss += loss.item()
            running_corrects += torch.sum(predicted == label.detach()).item()
            verbose_loss += loss.item()
            if i% 100 == 0 and i !=0:
                print("Loss over last 100 batches: " + str(verbose_loss/100))
                verbose_loss = 0.0


        # Verbose 2
        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects / train_size
        print(" ")
        print(" ")
        print(" ")
        print("****************")
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # Validation
        running_corrects = 0
        running_loss = 0.0
        verbose_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(validloader):
                input, label = data[0].to(device), data[1].to(device)
                output = model(input)
                _, predicted = torch.max(output, 1) # max dumped in _, argmax in predicted
                loss = loss_function(output, label)
                running_loss += loss.item()
                running_corrects += torch.sum(predicted == label.data).item()

            # Verbose 3
            epoch_loss = running_loss / valid_size
            epoch_acc = running_corrects / valid_size
            print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # Copy the model if it gets better
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model = copy.deepcopy(model.state_dict())
    # Verbose 4
    time_elapsed = time.time() - since
    print("Training finished in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    print("Best validation accuracy: " + str(best_acc))


    model.load_state_dict(best_model) # In place anyway
    return model # Returning just in case

##
rnn = RNN()
rnn.to(device)

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(rnn.parameters(), lr=0.000001, amsgrad=True)

trainModel(rnn, loss_function, optimizer, 1)

## Checkpoints
def makeCheckpoint(model, save_path):
    torch.save(model.state_dict(), save_path)
    print("Weights saved to: " + save_path)
    return

save_path = './weights/ONLY_LAST_OUTPUT-ResNet34&3LSTM(600)-6epochs.pth'
makeCheckpoint(rnn, save_path)


































