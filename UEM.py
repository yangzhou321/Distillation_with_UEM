import torch
import torch.nn as nn
import torch.nn.functional as F


class head_branch(nn.Module):
    def __init__(self, in_channels):
        super(head_branch, self).__init__()
        self.in_channels = in_channels
        self.theta = nn.Sequential(nn.Conv2d(in_channels, 512, 1),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, 3, 1, 1),
                                   # nn.BatchNorm2d(in_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(512, in_channels, 1),
                                   nn.ReLU()
                                   )

    def forward(self, x):
        theta = x + self.theta(x)
        return theta


class BottleNeck(nn.Module):
    def __init__(self, in_channel):
        super(BottleNeck, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, in_channel, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.conv(x)


class FE_Net(nn.Module):
    def __init__(self, in_channel, scale=2, btn=3):
        super(FE_Net, self).__init__()
        self.upsample = nn.Sequential(nn.ConvTranspose2d(in_channel, in_channel, 2, 2),
                                      nn.ReLU())

        layer = []
        for i in range(btn):
            layer.append(BottleNeck(in_channel))
        self.conv = nn.Sequential(*layer)
        self.mu = head_branch(in_channel)
        self.theta = head_branch(in_channel)
        # self.sw = nn.Sequential(nn.Conv2d(in_channel,1,3,1,1),
        #                         nn.ReLU(),
        #                         nn.AdaptiveAvgPool2d(output_size=(1,1)))
        self.downsample = nn.AvgPool2d(2, 2)

    def forward(self, x):
        mid = x + self.downsample(self.conv(self.upsample(x)))
        mu = self.mu(mid)
        # theta = self.sw(self.theta(mid))
        theta = self.theta(mid)
        return mu, -theta

