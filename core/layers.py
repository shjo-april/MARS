import torch
from torch import nn

from tools.ai import torch_utils

class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding=0, dilation=1, bias=False):
        super().__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, low_channels=48):
        super().__init__()
        
        self.low_block = ConvBlock(in_channels, low_channels, 1)
        
        self.classifier = nn.Sequential(
            ConvBlock(out_channels + low_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout(0.5),

            ConvBlock(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout(0.1),

            nn.Conv2d(out_channels, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        )
    
    def forward(self, x, x_low, with_features=False):
        x_low = self.low_block(x_low)

        x = torch_utils.resize(x, size=x_low.size()[2:])
        x = torch.cat((x, x_low), dim=1)

        x = self.classifier(x)

        if with_features:
            return x, x_low
        return x

class ASPP_For_DeepLabv3(nn.Module):
    def __init__(self, in_channels, out_channels, output_stride):
        super().__init__()
        
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        
        self.aspp1 = ConvBlock(in_channels, out_channels, 1, padding=0, dilation=dilations[0])
        self.aspp2 = ConvBlock(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ConvBlock(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ConvBlock(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBlock(in_channels, out_channels, 1)
        )
        
        self.block = nn.Sequential(
            ConvBlock(out_channels * 5, out_channels, 1),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x5 = self.global_avg_pool(x)
        x5 = torch_utils.resize(x5, size=x4.size()[2:])

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.block(x)

        return x
