import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

from utils import load_data_from_folder
from data import ImageDataset, transformations
from model import load_model, train_model, get_trainable


# Get data
folder_path = './data/images/*.jpg'
data = load_data_from_folder(folder_path)

# Set up dataset
train_transformations = [
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

test_transformations = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ]
    
train_size = 5542
test_size = 1848

train_transforms = transformations(train_transformations)
test_transforms = transformations(test_transformations)

initdataset = ImageDataset(data, transforms = None)

train_data, test_data = torch.utils.data.random_split(initdataset, [train_size, test_size])

train_data = ImageDataset(train_data, transforms=train_transforms)
test_data = ImageDataset(test_data, transforms=test_transforms)

train_dataloader = DataLoader(
    train_data, batch_size=64, shuffle = True, num_workers = 4
)

test_dataloader = DataLoader(
    test_data, batch_size=64, shuffle = True, num_workers = 4
)

# Load model
base_model = 'resnet18'
pretrained = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = load_model(base_model, device, pretrained, out_features = 37, layers_to_freeze = None, freeze_all = False)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(get_trainable(model.parameters()), lr=0.0001)

# Train Loop
n_epochs = 5
trained_model = train_model(model, optimizer, criterion, n_epochs, train_dataloader, test_dataloader, device)

# Load model from local


# Predict


