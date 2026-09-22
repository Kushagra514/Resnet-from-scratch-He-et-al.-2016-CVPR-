import csv 
import os 

import torch 
import torch.nn as nn
import torch.optim as optim 

from data import get_cifar10_loaders
from models import PlainCNN

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)


        #1. clear old gradients
        optimizer.zero_grad()

        #2. forward pass 
        outputs = model(images)

        #3. calculate loss
        loss = criterion(outputs,labels)

        #4. Backpropogation 
        loss.backward()

        #5. Update weights
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim = 1)

        correct +=( predictions == labels).sum().item()
        total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_accuracy = 100.0 * correct / total

        return epoch_loss, epoch_accuracy

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        predictions = output.argmax(dim = 1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss  / total
    epoch_accuracy = 100.0 * correct /  total

    return epoch_loss, epoch_accuaracy

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size = 128
    )

    model = PlainCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
            
    )