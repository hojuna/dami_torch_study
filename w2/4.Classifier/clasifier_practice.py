import torch
import torchvision
import torchvision.transforms as transforms
from net import Net

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn


def data_load(batch_size=4):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes, trainset, testset

# 이미지를 보여주기 위한 함수
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def display_data(trainloader,batch_size=4):
    # 학습용 이미지를 무작위로 가져오기
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # 이미지 보여주기
    imshow(torchvision.utils.make_grid(images))
    # 정답(label) 출력
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


def train(train_epoch=10,model=None):
    net = model if model is not None else Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    device = torch.device('cpu')
    net.to(device)

    for epoch in range(train_epoch):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # GPU로 데이터 이동
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

def save_model():
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

def load_model():
    net = Net()
    net.load_state_dict(torch.load(PATH))



def evaluate(net):
    device = torch.device('cpu')
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # GPU로 데이터 이동
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')


def evaluate_class(net):
    device = torch.device('cpu')
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # GPU로 데이터 이동
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # 각 분류별 정확도(accuracy)를 출력합니다
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


if __name__ == "__main__":

    net = Net()
    batch_size = 4
    trainloader, testloader, classes, trainset, testset = data_load(batch_size)
    display_data(trainloader,batch_size)
    train(5,net)
    # save_model()
    # load_model()
    evaluate(net)
    evaluate_class(net)

"""
evaluate 결과

Accuracy of the network on the 10000 test images: 60 %
Accuracy for class: plane is 57.5 %
Accuracy for class: car   is 72.8 %
Accuracy for class: bird  is 35.7 %
Accuracy for class: cat   is 27.9 %
Accuracy for class: deer  is 60.6 %
Accuracy for class: dog   is 59.7 %
Accuracy for class: frog  is 67.9 %
Accuracy for class: horse is 65.3 %
Accuracy for class: ship  is 79.9 %
Accuracy for class: truck is 77.7 %


"""