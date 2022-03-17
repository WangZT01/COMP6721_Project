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



def buildDatasets():

    transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    data_set = datasets.ImageFolder(root = '../Data/', transform = transform)
    n = len(data_set)
    n_test = int( n / 4 )
    n_train = n - n_test

    X_train, X_test = data.random_split(data_set, (n_train, n_test))
    y_train = np.array([y for x, y in iter(X_train)])
    y_test = np.array([y for x, y in iter(X_test)])
    return X_train, y_train, X_test, y_test

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=32, kernel_size=3, padding=1),
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
            nn.Linear(512, 5)
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)  # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)
        return x


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = buildDatasets()
    tic = time.time()
    torch.manual_seed(0)
    net = NeuralNetClassifier(
        CNN,
        max_epochs=1,
        iterator_train__num_workers=4,
        iterator_valid__num_workers=4,
        lr=1e-3,
        batch_size=64,
        optimizer=optim.Adam,
        criterion=nn.CrossEntropyLoss,
        device=torch.device("cpu")
    )
    net.fit(X_train, y = y_train)
    y_pred = net.predict(X_test)
    accuracy_score(y_test, y_pred)
    plot_confusion_matrix(net, X_test, y_test.reshape(-1, 1))
    plt.show()

    toc = time.time()
    print('duration=', toc - tic)

