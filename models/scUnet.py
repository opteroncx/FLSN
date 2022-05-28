# DL-SIM Nature communication2020
import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  if your machine do not have enough memory to handle all those weights
        #  bilinear interpolation could be used to do the upsampling.
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class Net(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(Net, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down11 = down(64, 128)
        self.down21 = down(128, 256)
        self.down31 = down(256, 512)
        self.down41 = down(512, 1024)
        self.up11 = up(1024, 512)
        self.up21 = up(512, 256)
        self.up31 = up(256, 128)
        self.up41 = up(128, 64)
        self.unet_1st_out = outconv(64, n_channels)
        
        self.inc0 = inconv(n_channels*2, 64)
        self.down12 = down(64, 128)
        self.down22 = down(128, 256)
        self.down32 = down(256, 512)
        self.down42 = down(512, 1024)
        self.up12 = up(1024, 512)
        self.up22 = up(512, 256)
        self.up32 = up(256, 128)
        self.up42 = up(128, 64)
        self.unet_2nd_out = outconv(64, n_classes)
        

    def forward(self, x):
        x_in = x
        x1 = self.inc(x)
        x2 = self.down11(x1)
        x3 = self.down21(x2)
        x4 = self.down31(x3)
        x5 = self.down41(x4)
        x = self.up11(x5, x4)
        x = self.up21(x, x3)
        x = self.up31(x, x2)
        x = self.up41(x, x1)
        x = self.unet_1st_out(x)
        x = torch.cat([x_in, x], dim=1)
        x1 = self.inc0(x)
        x2 = self.down12(x1)
        x3 = self.down22(x2)
        x4 = self.down32(x3)
        x5 = self.down42(x4)
        x = self.up12(x5, x4)
        x = self.up22(x, x3)
        x = self.up32(x, x2)
        x = self.up42(x, x1)
        x = self.unet_2nd_out(x)
        return x
