import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from data.mnist import load_mnist
from data.userdata import load_user_data
from models.simple_nn import SimpleNN

# Choose to load the MNIST dataset (True) or User Dataset (False)
MNIST = False

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
model.load_state_dict(torch.load('models/mnist_cnn.pt'))

# Load the dataset
if MNIST:
    train_loader, test_loader = load_mnist()
else:
    _, test_loader = load_user_data()

# Function to display an image
def display_image(image):
    image = image.squeeze()
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.show()

# Function to run the model on an image
def run_model(image):
    image = image.to(device)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# Main loop
repeat = True
while repeat:
    # Ask the user to pick an image
    if MNIST:
        index = int(input("Enter the index of the image (0-9999): "))
        image, label = test_loader.dataset[index]
    else:
        image, label = test_loader.dataset[0]
        repeat = False

    # Display the image
    display_image(image)

    # Run the image through the model
    predicted = run_model(image)

    # Print the result
    print(f"Predicted label: {predicted}")
    print(f"Actual label: {label}")