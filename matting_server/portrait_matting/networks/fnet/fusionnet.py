import torch
import torch.nn as nn

from portrait_matting.networks.ops import Conv2dIBNormRelu


class FusionNet(nn.Module):

    def __init__(self):
        super(FusionNet, self).__init__()
        self.conv1 = Conv2dIBNormRelu(7, 64, 5, stride=1, padding=2)
        self.conv2 = Conv2dIBNormRelu(64, 32, 3, stride=1, padding=1)
        self.conv3 = Conv2dIBNormRelu(32, 16, 3, stride=1, padding=1)
        self.conv4 = Conv2dIBNormRelu(16, 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self._init_norm(m)

    @staticmethod
    def _init_conv(conv):
        nn.init.kaiming_uniform_(
            conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    @staticmethod
    def _init_norm(norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        pred_matte = torch.sigmoid(x)

        return pred_matte
