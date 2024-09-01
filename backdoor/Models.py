import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.backdoor_trigger = torch.ones(1, 5, 5)

    def forward(self, inputs, activate_backdoor=1):
        # 确保输入是四维张量，以匹配卷积层的期望形状
        if inputs.dim() != 4 or inputs.size(1) != 1:
            inputs = inputs.view(inputs.size(0), 1, 28, 28)
        
        tensor = inputs  # 使用 inputs 作为 tensor 的初始状态

        if activate_backdoor:
            # 创建后门触发器，这里假设为5x5的全1矩阵
            trigger = torch.ones(1, 5, 5).to(inputs.device)
            # 将触发器添加到输入图像的中心位置
            trigger_location = ((28 - 5) // 2, (28 - 5) // 2)
            tensor[:, :, trigger_location[0]:trigger_location[0] + 5, trigger_location[1]:trigger_location[1] + 5] = \
                tensor[:, :, trigger_location[0]:trigger_location[0] + 5, trigger_location[1]:trigger_location[1] + 5] * (1 - trigger)

        # 执行正常的前向传播
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7*7*64)  # 展平特征以传递到全连接层
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)  # 应用全连接层

        return tensor


