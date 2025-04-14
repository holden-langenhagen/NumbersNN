import os
import torch
import torch.optim as optim
from data.mnist import load_mnist
from models.simple_nn import SimpleNN

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)  # flatten input data
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    torch.save(model.state_dict(), 'models/mnist_cnn.pt')
    print("Saved model in models/mnist_cnn.pt")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN().to(device)
    if os.path.exists('models/mnist_cnn.pt'):
        model.load_state_dict(torch.load('models/mnist_cnn.pt'))
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_loader, _ = load_mnist()
    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, epoch)

if __name__ == '__main__':
    print('Hi')
    main()