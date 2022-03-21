# %%
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F  # for using function relu()
import torch.optim as optim

# %%

"""
in_channels, out_channels = 5, 10
width, height = 100, 100           # pic size
kernel_size = 3
batch_size = 1

input = torch.rand(batch_size,
                   in_channels,                       # 只能设置为5
                   width,
                   height)

conv_layer = torch.nn.Conv2d(in_channels,             # 卷积层设置
                             out_channels,
                             kernel_size=kernel_size)

output = conv_layer(input)


例子
input = [1,2,3,4,5
         1,2,3,4,5
         1,2,3,4,5
         1,2,3,4,5
         1,2,3,4,5]

input = torch.Tensor(input).view(1,1,5,5)  # Batch, Channel, W, H

conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1，stride=2 , bias=False)

kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3)

conv_layer.weight.data = kernel.data  # 初始化核


maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)  # default value 


"""



# %%

batch_size = 1

# using dataset MNIST: 28*28 per pic
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
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=(5, 5))
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

# 使用激活函数 ReLu
    def forward(self, x):
        # 用图像的维度作为batch——size
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  # flatten
        print(x.size())
        x = self.fc(x)
        return x      # 不用激活


model = Net()

""""
用GPU操作
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

"""

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
for epoch in range(1):
    train(epoch)
    test()

"""
[1,   300] loss: 0.654
[1,   600] loss: 0.219
[1,   900] loss: 0.153
[1,  1200] loss: 0.152
[1,  1500] loss: 0.126
[1,  1800] loss: 0.122
[1,  2100] loss: 0.104
[1,  2400] loss: 0.099
[1,  2700] loss: 0.085
[1,  3000] loss: 0.089
Accuracy on test set: 97 %
[2,   300] loss: 0.071
[2,   600] loss: 0.077
[2,   900] loss: 0.079
[2,  1200] loss: 0.067
[2,  1500] loss: 0.065
[2,  1800] loss: 0.071
[2,  2100] loss: 0.051
[2,  2400] loss: 0.072
[2,  2700] loss: 0.056
[2,  3000] loss: 0.054
Accuracy on test set: 98 %
[3,   300] loss: 0.043
[3,   600] loss: 0.060
[3,   900] loss: 0.058
[3,  1200] loss: 0.045
[3,  1500] loss: 0.054
[3,  1800] loss: 0.045
[3,  2100] loss: 0.043
[3,  2400] loss: 0.051
[3,  2700] loss: 0.047
[3,  3000] loss: 0.050
Accuracy on test set: 98 %
"""