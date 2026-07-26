# computation stuff
import numpy as np
import torch
from torch import nn

# class MyModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         # initialize everything from the root class
#         # PyTorch won't allow you to define variable using nn.LayerName before this call
#         super().__init__()

#         self.net1 = nn.Linear(input_size, hidden_size)
#         self.net2 = nn.ReLU()
#         self.net3 = nn.Linear(hidden_size, hidden_size)
#         self.net4 = nn.ReLU()
#         self.net5 = nn.Linear(hidden_size, output_size)

#     def forward(self, input_data):
#         x = self.net1(input_data)
#         x = self.net2(x)
#         x = self.net3(x)
#         x = self.net4(x)
#         output = self.net5(x)
#         return output


# the Max-Feature-Map activation (MFM) which is based on Max-Out activation function
class MFM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.max(x[:, : (x.size(1) // 2)], x[:, (x.size(1) // 2) :])


class LCNN(nn.Module):
    # in conv we will create new priznaki
    # in mfm we will activate sampling, i mean, on this stage we choice the best featurs and from all features
    # we will lost only these best
    # in batchnorm we will normalize our dataset i mean we will use it for avoiding deleysin training and retraining too
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()

        # Conv 1          5 × 5 / 1 × 1      863 × 600 × 64       1.6K
        # MFM 2            −                 864 × 600 × 32        −
        # MaxPool3       2 × 2 / 2 × 2     431 × 300 × 32         −

        #                                           |      boardings   |
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.mfm1 = MFM()  # <- there is output is 32
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv4           1 × 1 / 1 × 1     431 × 300 × 64       2.1K
        # MFM 5             −               431 × 300 × 32         −
        # BatchNorm 6       −               431 × 300 × 32         −

        # after mfm1 ounput 32, therefore for conv2 input will be 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1)
        self.mfm2 = MFM()
        self.bn1 = nn.BatchNorm2d(32)

        # Conv 7          3 × 3 / 1 × 1      431 × 300 × 96      27.7K
        # MFM8               −               431 × 300 × 48        −
        # MaxPool 9       2 × 2 / 2 × 2      215 × 150 × 48        −
        # BatchNorm 10       −               215 × 150 × 48        −
        self.conv3 = nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1)
        self.mfm3 = MFM()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(48)

        # Conv 11          1 × 1 / 1 × 1     215 × 150 × 96      4.7K
        # MFM 12             −               215 × 150 × 48        −
        # BatchNorm 13       −               215 × 150 × 48        −
        self.conv4 = nn.Conv2d(48, 96, kernel_size=1)
        self.mfm4 = MFM()
        self.bn3 = nn.BatchNorm2d(48)

        # Conv 14          3 × 3 / 1 × 1     215 × 150 × 128     55.4K
        # MFM 15             −               215 × 150 × 64        −
        # MaxPool 16       2 × 2 / 2 × 2     107 × 75 × 64         −
        self.conv5 = nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1)
        self.mfm5 = MFM()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv6 = nn.Conv2d(64, 128, kernel_size=1)
        self.mfm6 = MFM()
        self.bn4 = nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.mfm7 = MFM()
        self.bn5 = nn.BatchNorm2d(32)

        self.conv8 = nn.Conv2d(32, 64, kernel_size=1)
        self.mfm8 = MFM()
        self.bn6 = nn.BatchNorm2d(32)

        self.conv9 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.mfm9 = MFM()
        self.pool4 = nn.MaxPool2d(2, 2)

        # for final forcase
        # linear(in_features, out_features)
        # And dropout 0.75 was used to reduce overfitting
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.75)
        self.fc1 = nn.Linear(32, 160)
        self.mfm10 = MFM()
        self.bn7 = nn.BatchNorm1d(80)
        self.fc2 = nn.Linear(80, num_classes)

    def forward(self, x, **kwargs):
        if isinstance(x, dict):
            x = x["data_object"]

        x = self.conv1(x)
        x = self.mfm1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.mfm2(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = self.mfm3(x)
        x = self.pool2(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = self.mfm4(x)
        x = self.bn3(x)

        x = self.conv5(x)
        x = self.mfm5(x)
        x = self.pool3(x)

        x = self.conv6(x)
        x = self.mfm6(x)
        x = self.bn4(x)

        x = self.conv7(x)
        x = self.mfm7(x)
        x = self.bn5(x)

        x = self.conv8(x)
        x = self.mfm8(x)
        x = self.bn6(x)

        x = self.conv9(x)
        x = self.mfm9(x)
        x = self.pool4(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.mfm10(x)
        x = self.bn7(x)
        x = self.fc2(x)

        return x
