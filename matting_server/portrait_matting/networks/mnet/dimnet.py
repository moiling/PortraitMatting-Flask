import torch.nn as nn


class Conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride,
                 padding, bias=True, dilation=1, with_bn=True, with_relu=True):

        super().__init__()

        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                             padding=padding, stride=stride, bias=bias, dilation=dilation)

        if with_bn:
            if with_relu:
                self.cbr_unit = nn.Sequential(
                    conv_mod,
                    nn.BatchNorm2d(int(n_filters)),
                    nn.ReLU(inplace=True)
                )
            else:
                self.cbr_unit = nn.Sequential(
                    conv_mod,
                    nn.BatchNorm2d(int(n_filters))
                )
        else:
            if with_relu:
                self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))
            else:
                self.cbr_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class SegNetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv1 = Conv2DBatchNormRelu(in_size, out_size, k_size=3, stride=1, padding=1)
        self.conv2 = Conv2DBatchNormRelu(out_size, out_size, k_size=3, stride=1, padding=1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class SegNetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv1 = Conv2DBatchNormRelu(in_size, out_size, k_size=3, stride=1, padding=1)
        self.conv2 = Conv2DBatchNormRelu(out_size, out_size, k_size=3, stride=1, padding=1)
        self.conv3 = Conv2DBatchNormRelu(out_size, out_size, k_size=3, stride=1, padding=1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs, skip_maxpool=False):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        maxpool_outputs, indices = self.maxpool_with_argmax(outputs)
        if skip_maxpool:
            return outputs, indices, unpooled_shape
        return maxpool_outputs, indices, unpooled_shape


class SegNetUp1(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv = Conv2DBatchNormRelu(in_size, out_size, k_size=5, stride=1, padding=2, with_relu=False)

    def forward(self, inputs, indices, output_shape, skip_unpool=False):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        if skip_unpool:
            outputs = inputs
        outputs = self.conv(outputs)
        return outputs


class DIMNet(nn.Module):
    def __init__(self, n_classes=1, in_channels=6, is_unpooling=True):
        super().__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = SegNetDown2(self.in_channels, 64)
        self.down2 = SegNetDown2(64, 128)
        self.down3 = SegNetDown3(128, 256)
        self.down4 = SegNetDown3(256, 512)
        self.down5 = SegNetDown3(512, 512)

        self.up5 = SegNetUp1(512, 512)
        self.up4 = SegNetUp1(512, 256)
        self.up3 = SegNetUp1(256, 128)
        self.up2 = SegNetUp1(128, 64)
        self.up1 = SegNetUp1(64, n_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # inputs: [N, 4, 320, 320]
        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        # down5, indices_5, unpool_shape5 = self.down5(down4)
        down5, _, _ = self.down5(down4, skip_maxpool=True)

        # up5 = self.up5(down5, indices_5, unpool_shape5)
        up5 = self.up5(down5, indices_4, unpool_shape4, skip_unpool=True)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        # x = torch.squeeze(up1, dim=1)  # [N, 1, 320, 320] -> [N, 320, 320]
        out = self.sigmoid(up1)

        return out
