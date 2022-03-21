# %%
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F  # for using function relu()
import torch.optim as optim

# %%
# Data preparation
import CNN

batch_size = 20

# using dataset MNIST: 28*28 per pic
transform = transforms.Compose([transforms.ToTensor(),  # Convert the PIL Image to Tensor.
                                transforms.Normalize((0.1307,), (0.3081,))])  # 归一化为张量 C*W*H 「0-1」；整个的均值、方差

# 1. prepare dataset
# 以下为训练及测试的数据集
train_dataset = datasets.MNIST(root='/Users/star/Desktop/coding/Basic/Softmax',
                               train=True,
                               download=True,
                               transform=transform)  # 会直接用上面定义的transform进行处理

train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_dataset = datasets.MNIST(root='/Users/star/Desktop/coding/Basic/Softmax',
                              train=True,
                              download=True,
                              transform=transform)

test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


# %%
class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=(1, 1))
        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=(1, 1))

        self.branch5x5 = torch.nn.Conv2d(16, 24, kernel_size=(5, 5), padding=2)

        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=(3, 3), padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch1x1(x)
        branch5x5 = self.branch5x5(branch5x5)

        branch3x3 = self.branch1x1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=(3, 3), stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


class GoogLeNet(torch.nn.Module):
    # 五层网络
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=(5, 5))

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(1408, 10)

    # 使用激活函数 ReLu
    def forward(self, x):
        # 用图像的维度作为batch——size
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)  # flatten
        x = self.fc(x)
        return x  # 不用激活


model = GoogLeNet()


class ResBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.channels = channels

        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        y = F.relu(self.conv1)
        y= self.conv1(y)
        return F.relu(x+y)



# %%
# Construct loss and optimizer
# 交叉熵损失
criterion = torch.nn.CrossEntropyLoss()

# 带加速度的优化器
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# %%
# 3.train and test
# 将训练和测试封装在一个函数之中
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        # inputs, target = inputs.to(device),target.to(device)
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0

    # 不会计算梯度
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # images, labels = images.to(device),labels.to(device)
            outputs = model(images)

            # 沿着每一行找最大值的下标即其数值
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy on test set: %d %%' % (100 * correct / total))


# %%
for epoch in range(3):
    train(epoch)
    test()

"""
[1,   300] loss: 0.895
[1,   600] loss: 0.264
[1,   900] loss: 0.159
[1,  1200] loss: 0.146
[1,  1500] loss: 0.141
[1,  1800] loss: 0.117
[1,  2100] loss: 0.108
[1,  2400] loss: 0.095
[1,  2700] loss: 0.098
[1,  3000] loss: 0.094
Accuracy on test set: 97 %
[2,   300] loss: 0.066
[2,   600] loss: 0.071
[2,   900] loss: 0.083
[2,  1200] loss: 0.070
[2,  1500] loss: 0.080
[2,  1800] loss: 0.062
[2,  2100] loss: 0.069
[2,  2400] loss: 0.063
[2,  2700] loss: 0.063
[2,  3000] loss: 0.061
Accuracy on test set: 98 %
"""