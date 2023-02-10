import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

from utils import load_data_from_folder
from data import ImageDataset, transformations


# Get data
folder_path = ''
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

# Set up loss function and optimizer

# Train Loop


# Load trained model

# Predict


