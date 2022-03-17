import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td


def cifar_loader(batch_size, shuffle_test=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./data', train=True, download=True,
                 transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, 4), transforms.ToTensor(),normalize]))
    test = datasets.CIFAR10('./data', train=False, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                 shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader


batch_size = 64
test_batch_size = 64
input_size = 3072
N = batch_size
D_in = input_size
H = 50
D_out = 10
num_epochs = 10
train_loader, _ = cifar_loader(batch_size)
_, test_loader = cifar_loader(test_batch_size)

#defining the network
class MultiLayerFCNet(nn.Module):
    # design
    def __init__(self, D_in, H, D_out):
        super(MultiLayerFCNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, D_out)
    
    # DATA FLOW
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return F.log_softmax(x,dim=1)
    
model = MultiLayerFCNet(D_in, H, D_out)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

#Train the model
for epoch in range(num_epochs):
    avg_loss_epoch = 0
    batch_loss = 0
    total_batches = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 32*32 * 3)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_batches += 1
        batch_loss += loss.item()
        # print(" i = ", i)
    # print("epoch = ", epoch)

    avg_loss_epoch = batch_loss / total_batches

correct = 0
total = 0
for images, labels in test_loader:
    images = images.reshape(-1, 3 * 32 * 32)
    outputs_test = model(images)
    _, predicted = torch.max(outputs_test.data, 1)
    total += labels.size(0)

    correct += (predicted == labels).sum().item()

print('Accury of the network on the 1000 test images:%d %%' % (100 * correct / total ))
