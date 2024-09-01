import torch
import torch.nn as nn
import torch.nn.functional as F


class Cifar10_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 根据灰度图像调整第一个全连接层的输入特征数
        self.fc1 = nn.Linear(64 * (32 // 4) * (32 // 4), 512)  # 32 // 4 等于 8，假设每次池化后尺寸减半
        self.fc2 = nn.Linear(512, 10)

    def forward(self, inputs):
        tensor = inputs  # 直接使用输入
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        # 正确计算展平操作后的尺寸
        tensor = tensor.view(-1, 64 * (32 // 4) * (32 // 4))  # 调整为正确的尺寸
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor

    
# class Cifar10_CNN(nn.Module):
#     def __init__(self):
#         super(Cifar10_CNN, self).__init__()
#         # 第一个卷积层
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
#         # 池化层
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         # 第二个卷积层
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
#         # 第三个卷积层，输出通道数为64
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        
#         # 全连接层
#         # 根据CIFAR-10数据集图像尺寸和卷积层输出计算全连接层的输入特征数
#         self.fc1 = nn.Linear(64 * 8 * 8, 512)  # 假设conv3输出尺寸为8x8
#         self.fc2 = nn.Linear(512, 10)  # CIFAR-10有10个类别

#     def forward(self, inputs):
#         tensor = inputs
#         # 应用第一个卷积层，激活函数，然后池化
#         tensor = self.pool(F.relu(self.conv1(tensor)))
#         # 应用第二个卷积层，激活函数，然后池化
#         tensor = self.pool(F.relu(self.conv2(tensor)))
#         # 应用第三个卷积层和激活函数
#         tensor = F.relu(self.conv3(tensor))
        
#         # 展平特征图以匹配全连接层的输入
#         tensor = tensor.view(-1, 64 * 8 * 8)  # 根据conv3的输出尺寸进行调整
#         # 应用第一个全连接层和激活函数
#         tensor = F.relu(self.fc1(tensor))
#         # 应用第二个全连接层，输出原始类别分数
#         tensor = self.fc2(tensor)
#         return tensor
    
class Mnist_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)
        return tensor

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7*7*64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor
# RESNET20网络模型
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差连接需要先调整输入特征，如果输入和输出通道数不一致
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)  # 计算快捷通路
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # 添加残差
        out = self.relu(out)
        return out

# 定义ResNet20模型
class ResNet20(nn.Module):
    def __init__(self):
        super(ResNet20, self).__init__()
        self.in_channels = 16  # 输入通道数为16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, 16, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, num_blocks=3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, num_blocks=3, stride=2)
        self.fc = nn.Linear(64 * BasicBlock.expansion, 10)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.avg_pool2d(x, 8)  # Global Average Pooling
        x = x.view(x.size(0), -1)  # Flatten for the classifier
        x = self.fc(x)
        return x