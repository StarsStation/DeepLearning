# %%
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F  # for using function relu()
import torch.optim as optim
import matplotlib.pyplot as plt


# %%

# using dataset MNIST: 28*28 per pic

batch_size = 64

transform = transforms.Compose([transforms.ToTensor(),                        # Convert the PIL Image to Tensor.
                                transforms.Normalize((0.1307,), (0.3081,))])  # 归一化为张量 C*W*H 「0-1」；整个的均值、方差

# 1. prepare dataset
# 以下为训练及测试的数据集
train_dataset = datasets.MNIST(root='/Users/star/Desktop/coding/Basic/Softmax',
                               train=True,
                               download=True,
                               transform=transform) # 会直接用上面定义的transform进行处理

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
# 2.design model
class Net(torch.nn.Module):

# 五层网络
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

# 使用激活函数 ReLu
    def forward(self, x):
        # 将像素信息变为行向量
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)      # 不用激活


model = Net()

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
            outputs = model(images)

            # 沿着每一行找最大值的下标即其数值
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy on test set: %d %%' % (100 * correct / total))


# %%
for epoch in range(10):
    train(epoch)
    test()
