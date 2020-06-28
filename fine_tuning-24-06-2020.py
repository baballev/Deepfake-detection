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
"""
Fine tuning ResNet 101 on 2 classes: DeepFake / NonDeepFake
"""

## GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Corrupted files check
def isFileNotCorrupted(path):
    return not(os.stat(path).st_size <= 50) # If file size is inferior to 50 bytes, prune it

## Loading dataset
train_path = 'E:/Programmation/Python/PAF 2020/deepfake2/dataset-paf/tmp/train/'
test_path = 'E:/Programmation/Python/PAF 2020/deepfake2/dataset-paf/tmp/test/'

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder(train_path, transform=transform, is_valid_file=isFileNotCorrupted)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)

testset = torchvision.datasets.ImageFolder(test_path, transform=transform, is_valid_file=isFileNotCorrupted)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

train_size = len(trainset)
test_size = len(testset)

##  Training
# TODO: Automatiser un process de check de stagnation de la loss pour éviter l'over-fitting

# /!\ This is not fine tuning, only retraining ResNet101 from its trained weights on ImageNet
def train_model(model, loss_function, optimizer, scheduler, epochs_nb):
    since = time.time()

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs_nb):
        # Verbose 1
        print("Epoch [" + str(epoch+1) + " / " + str(epochs_nb) + "]")
        print("-" * 10)

        # Training
        running_corrects = 0
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            input, label = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            output = classifier(input)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output, 1) # max pruned in _, argmax in predicted
            running_loss += loss.item()
            running_corrects += torch.sum(predicted == label.data).item()

        scheduler.step()

        # Verbose 2
        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects / train_size
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # Validation
        running_corrects = 0
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                input, label = data[0].to(device), data[1].to(device)
                output = model(input)
                _, predicted = torch.max(output, 1) # max pruned in _, argmax in predicted
                loss = loss_function(output, label)
                running_loss += loss.item()
                running_corrects += torch.sum(predicted == label.data).item()

            # Verbose 3
            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects / test_size
            print(running_corrects)
            print(test_size)
            print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # Copy the model if it gets better
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print("Training finished in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))

    model.load_state_dict(best_model)
    return model

## Checkpoint
def makeCheckpoint(model, save_path):
    torch.save(model.state_dict(), save_path)
    print("Weights saved to: " + save_path)
    return

def loadCheckpoint(load_path):
    classifier = torchvision.models.resnet101(pretrained=False)
    num_features = classifier.fc.in_features
    classifier.fc = nn.Linear(num_features, 2)
    classifier.load_state_dict(torch.load(load_path))
    return classifier

## Other Hyperparameters
classifier = torchvision.models.resnet101(pretrained=True)

num_features = classifier.fc.in_features
classifier.fc = nn.Linear(num_features, 2) # Here we change the last layer from 2048, 1000 to 2048,2
print(classifier)
classifier.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) # Multiply learning rate by 0.1 every 3 epochs

# Summary
print(classifier)

save_path = 'E:/Programmation/Python/PAF 2020/deepfake2/weights/24-06-2020-ResNet101-5epochs.pth'
load_path = 'E:/Programmation/Python/PAF 2020/deepfake2/weights/24-06-2020-ResNet101-5epochs-acc=0.98.pth'

train_model(classifier, loss_function, optimizer, exp_lr_scheduler, 5)

makeCheckpoint(classifier, save_path)

classifier = loadCheckpoint(load_path)
# TODO: Fine-Tune instead of training the whole network
# ToDo: Enlarge the dataset, especially fake class















