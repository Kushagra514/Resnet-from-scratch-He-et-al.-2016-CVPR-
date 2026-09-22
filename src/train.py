import csv 
import os 

import torch 
import torch.nn as nn
import torch.optim as optim 

from data import get_cifar10_loaders
from models import PlainCNN

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()