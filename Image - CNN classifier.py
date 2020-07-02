import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

## GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


## NN
class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.conv1 = nn.Conv2d(3, 10, 5) # IN: [256,256,3] OUT: [252,252,10]
        self.conv2 = nn.Conv2d(10, 24, 3) # IN: [126,126,10] OUT: [124,124,24]
        self.conv3 = nn.Conv2d(24, 32, 3) # IN: [62,62,24] OUT: [60,60,32]

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.AvgPool2d(2)

        self.fc1 = nn.Linear(60*60*32, 1280)
        self.fc2 = nn.Linear(1280, 640)
        self.fc3 = nn.Linear(640, 240)
        self.fc4 = nn.Linear(240, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 60*60*32)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def isFileNotCorrupted(path):

    return not(os.stat(path).st_size <= 50) # If file size is inferior to 50 bytes, prune it





## Loading dataset
train_path = 'E:/Programmation/Python/PAF 2020/deepfake2/dataset-paf/v0/train/'
test_path = 'E:/Programmation/Python/PAF 2020/deepfake2/dataset-paf/v0/test/'

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder(train_path, transform=transform, is_valid_file=isFileNotCorrupted)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)

testset = torchvision.datasets.ImageFolder(test_path, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

## Other Hyperparameters
classifier = Classifier()
classifier.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)


##  Training
# Summary
print(classifier)

epochs_nb = 10
it_per_verbose = 25

for epoch in range(epochs_nb):
    running_loss = 0.0
    for i, data in enumerate(trainloader):

        input, label = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        output = classifier(input)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()

        # Verbose
        running_loss += loss.item()
        if i%it_per_verbose == (it_per_verbose - 1):
            print("Epoch [" + str(epoch) + "] - Iteration [" + str(i) + "] - Current loss = [" + str(running_loss/it_per_verbose) + "]")
            running_loss = 0.0


print("Training finished")

## Checkpoint
save_path = 'E:/Programmation/Python/PAF 2020/deepfake2/weights/23-06-2020-5epochs.pth'
torch.save(classifier.state_dict(), save_path)
print("Weights saved to: " + save_path)

## Tests
load_path = save_path

classifier = Classifier()
classifier.load_state_dict(torch.load(load_path))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, label = data
        out = classifier(images)
        _, predicted = torch.max(out, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

print("Accuracy on 250 images: " + str(100*(correct/total)) + "%.")

















