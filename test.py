import torch
from data.mnist import load_mnist
from models.simple_nn import SimpleNN

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # flatten input data
            test_loss += torch.nn.CrossEntropyLoss()(output, target).item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN().to(device)
    model.load_state_dict(torch.load('models/mnist_cnn.pt'))
    _, test_loader = load_mnist()
    test(model, device, test_loader)

if __name__ == '__main__':
    main()