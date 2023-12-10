# Libraries
import torch
import torch.nn as nn


# Helper classes/functions
class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.ac = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.ac(x)
        return x


def darknet_init():
    darknet = nn.Sequential(
        # First Layer
        # in_channels, kernel_number, kernel_size, stride, padding
        # (optional) modify the first argument from 3 to the image's true channel number
        CNN(3, 64, 7, 2, 3),
        nn.MaxPool2d(2, 2),

        # Second Layer
        CNN(64, 192, 3, 1, 1),
        nn.MaxPool2d(2, 2),

        # Third Layer
        CNN(192, 128, 1, 1, 0),
        CNN(128, 256, 3, 1, 1),
        CNN(256, 256, 1, 1, 0),
        CNN(256, 512, 3, 1, 1),
        nn.MaxPool2d(2, 2),

        # Fourth Layer
        CNN(512, 256, 1, 1, 0),
        CNN(256, 512, 3, 1, 1),
        CNN(512, 256, 1, 1, 0),
        CNN(256, 512, 3, 1, 1),
        CNN(512, 256, 1, 1, 0),
        CNN(256, 512, 3, 1, 1),
        CNN(512, 256, 1, 1, 0),
        CNN(256, 512, 3, 1, 1),
        CNN(512, 512, 1, 1, 0),
        CNN(512, 1024, 3, 1, 1),
        nn.MaxPool2d(2, 2),

        # Fifth Layer
        CNN(1024, 512, 1, 1, 0),
        CNN(512, 1024, 3, 1, 1),
        CNN(1024, 512, 1, 1, 0),
        CNN(512, 1024, 3, 1, 1),
        CNN(1024, 1024, 3, 1, 1),
        CNN(1024, 1024, 3, 2, 1),

        # Sixth Layer
        CNN(1024, 1024, 3, 1, 1),
        CNN(1024, 1024, 3, 1, 1)
    )
    return darknet


def dense_init():
    dense = nn.Sequential(
        nn.Linear(7 * 7 * 1024, 4096),
        nn.Dropout(0.5),
        nn.LeakyReLU(0.1),
        nn.Linear(4096, 7 * 7 * 30)
    )
    return dense


# Architecture
class YOLO(nn.Module):
    # input image has an expectation of channels of 3, modify darknet_init otherwise
    def __init__(self):
        super(YOLO, self).__init__()
        self.darknet = darknet_init()
        self.dense = dense_init()

    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)  # dim 0 is the # of samples
        x = self.dense(x)
        return x
