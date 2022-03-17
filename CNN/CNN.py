import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from skorch import NeuralNetClassifier
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils import data

num_epochs = 4
num_classes = 10
learning_rate = 0.001

def buildDatasets():

    transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    data_set = datasets.ImageFolder(root = '../Data/', transform = transform)
    n = len(data_set)
    n_test = int( n / 4 )
    n_train = n - n_test

    X_train, X_test = data.random_split(data_set, (n_train, n_test))
    y_train = np.array([y for x, y in iter(X_train)])
    y_test = np.array([y for x, y in iter(X_test)])

    train_loader = torch.utils.data.DataLoader(X_train, batch_size = 128, shuffle = True)
    test_loader = torch.utils.data.DataLoader(X_test, batch_size = 128, shuffle = True)

    return train_loader, y_train, test_loader, y_test

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4 * 224 * 224, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)  # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)
        return x

if __name__ == '__main__':

    tic = time.time()
    train_loader, y_train, test_loader, y_test = buildDatasets()
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Train accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))

    model.eval()
    with torch.no_grad():
        correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

    toc = time.time()
    print('duration=', toc - tic)