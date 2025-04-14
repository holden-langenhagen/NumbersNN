import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os

IMAGE_PATH = 'data/image.png'

class UserData(Dataset):
    def __init__(self, image, label):
        self.image = image
        self.label = label

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.image, self.label

def load_user_data():
    # Load the image from the file
    image = plt.imread(IMAGE_PATH)
    image = 1 - torch.tensor(image).mean(dim=2).reshape(1, 1, 28, 28).float()

    # Get the label from the user
    label = int(input("Enter the label for the image (0-9): "))

    # Return the image and label as tensors
    image = image.view(1, 1, 28, 28).float()

    dataset = UserData(image, label)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader