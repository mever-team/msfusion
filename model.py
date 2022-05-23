import math 
import torch.nn.functional as F
import numpy as np
from torch import nn
import torch 


class Conv2dSamePadding(nn.Conv2d):
    """2D Convolutions with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, name=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, groups=groups,
                         bias=bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        self.name = name

    def forward(self, x):
        input_h, input_w = x.size()[2:]
        kernel_h, kernel_w = self.weight.size()[2:]
        stride_h, stride_w = self.stride
        output_h, output_w = math.ceil(input_h / stride_h), math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * self.stride[0] + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * self.stride[1] + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.conv = Conv2dSamePadding(3, 32, 3)
        
        self.conv1 = Conv2dSamePadding(32, 64, 3)
        self.conv2 = Conv2dSamePadding(64, 128, 3)
        self.conv3 = Conv2dSamePadding(128, 256, 3)
        self.conv4 = Conv2dSamePadding(256, 512, 3)
        
        self.conv_last = Conv2dSamePadding(512, 32, 3)
        
        self.conv_skip = Conv2dSamePadding(64, 2, 1)
        
        self.max_pool = nn.MaxPool2d(2, 2)
        
        self.block1 = nn.Sequential(
            Conv2dSamePadding(32, 32, 3),
            nn.BatchNorm2d(32),
            Conv2dSamePadding(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.block2 = nn.Sequential(
            Conv2dSamePadding(64, 64, 3),
            nn.BatchNorm2d(64),
            Conv2dSamePadding(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.block3 = nn.Sequential(
            Conv2dSamePadding(128, 128, 3),
            nn.BatchNorm2d(128),
            Conv2dSamePadding(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.block4 = nn.Sequential(
            Conv2dSamePadding(256, 256, 3),
            nn.BatchNorm2d(256),
            Conv2dSamePadding(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.block5 = nn.Sequential(
            Conv2dSamePadding(512, 512, 3),
            nn.BatchNorm2d(512),
            Conv2dSamePadding(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.conv(x)
        residual1 = x
        x = self.block1(x)
        x =x+ residual1
        x = self.max_pool(x)
        
        x = self.conv1(x)
        residual2 = x
        x = self.block2(x)
        x =x+ residual2
        
        x_skip = self.conv_skip(x)
        
        x = self.max_pool(x)
        
        x = self.conv2(x)
        residual3 = x
        x = self.block3(x)
        x =x+ residual3
        x = self.max_pool(x)
        
        x = self.conv3(x)
        residual4 = x
        x = self.block4(x)
        x =x+ residual4
        x = self.max_pool(x)
        
        x = self.conv4(x)
        residual5 = x
        x = self.block5(x)
        x =x+ residual5
        x = self.max_pool(x)
        
        x = self.conv_last(x)
        
        return x, x_skip
    
class Encoder_no_skip(nn.Module):
    def __init__(self):
        super(Encoder_no_skip, self).__init__()
        
        self.conv = Conv2dSamePadding(3, 32, 3)
        
        self.conv1 = Conv2dSamePadding(32, 64, 3)
        self.conv2 = Conv2dSamePadding(64, 128, 3)
        self.conv3 = Conv2dSamePadding(128, 256, 3)
        self.conv4 = Conv2dSamePadding(256, 512, 3)
        
        self.conv_last = Conv2dSamePadding(512, 32, 3)
        
        self.max_pool = nn.MaxPool2d(2, 2)
        
        self.block1 = nn.Sequential(
            Conv2dSamePadding(32, 32, 3),
            nn.BatchNorm2d(32),
            Conv2dSamePadding(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.block2 = nn.Sequential(
            Conv2dSamePadding(64, 64, 3),
            nn.BatchNorm2d(64),
            Conv2dSamePadding(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.block3 = nn.Sequential(
            Conv2dSamePadding(128, 128, 3),
            nn.BatchNorm2d(128),
            Conv2dSamePadding(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.block4 = nn.Sequential(
            Conv2dSamePadding(256, 256, 3),
            nn.BatchNorm2d(256),
            Conv2dSamePadding(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.block5 = nn.Sequential(
            Conv2dSamePadding(512, 512, 3),
            nn.BatchNorm2d(512),
            Conv2dSamePadding(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.conv(x)
        residual1 = x
        x = self.block1(x)
        x =x+ residual1
        x = self.max_pool(x)
        
        x = self.conv1(x)
        residual2 = x
        x = self.block2(x)
        x =x+ residual2
        
        x = self.max_pool(x)
        
        x = self.conv2(x)
        residual3 = x
        x = self.block3(x)
        x =x+ residual3
        x = self.max_pool(x)
        
        x = self.conv3(x)
        residual4 = x
        x = self.block4(x)
        x =x+ residual4
        x = self.max_pool(x)
        
        x = self.conv4(x)
        residual5 = x
        x = self.block5(x)
        x =x+ residual5
        x = self.max_pool(x)
        
        x = self.conv_last(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder_dct, self).__init__()
        self.sb_encoder = Encoder_no_skip()
        
        self.encoder = Encoder()
        
        self.tran_conv1 = nn.ConvTranspose2d(64 , 32, kernel_size=2, stride=2,padding=0)
        self.tran_conv2 = nn.ConvTranspose2d(32 , 16, kernel_size=2, stride=2)
        self.tran_conv3 = nn.ConvTranspose2d(16 , 8, kernel_size=2, stride=2)
        self.tran_conv4 = nn.ConvTranspose2d(8 , 2, kernel_size=2, stride=2)
        self.tran_conv5 = nn.ConvTranspose2d(4 , 2, kernel_size=2, stride=2)
        self.dconv1 = DoubleConv(32,32)
        self.dconv2 = DoubleConv(16,16)
        self.dconv3 = DoubleConv(8,8)
        self.dconv4 = DoubleConv(2,2)
        
        self.conv = nn.Conv2d(2,1,kernel_size=1)
        
    def forward(self, x, xl):
        
        xd = self.sb_encoder(xl)

        x1, x2 = self.encoder(x)
        
        x = torch.cat((x1, xd), dim = 1)
        x = self.tran_conv1(x)
        x = self.tran_conv2(x)
        x = self.tran_conv3(x)
        x = self.tran_conv4(x)
        x = torch.cat((x, x2), dim = 1)
        x = self.tran_conv5(x)
        x = self.dconv4(x)
        x = self.conv(x)
        
        
        x = nn.AvgPool2d(3, stride=1, padding=1)(x)
        x = nn.Sigmoid()(x)
        x = torch.squeeze(x , 1)
        return x