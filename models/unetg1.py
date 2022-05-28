import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import CARB

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            CARB(64)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode='bilinear'):
        super().__init__()
        # 加全局残差
        if up_mode == 'conv':
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = CARB(64)
        else:
            self.up = nn.Upsample(scale_factor=2, mode=up_mode, align_corners=True)
            self.conv = nn.Sequential(
                        nn.Conv2d(128, 64, kernel_size=3, padding=1),
                        CARB(64)
                        )


    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Net(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        mode = 'bilinear'

        self.sfe = DoubleConv(n_channels, 64)
        self.sfeg = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 64)
        self.down2 = Down(64, 64)
        self.down3 = Down(64, 64)
        self.down4 = Down(64, 64)

        self.dg1 = Down(64, 64)
        self.dg2 = Down(64, 64)
        self.dg3 = Down(64, 64)
        self.dg4 = Down(64, 64)

        self.up1 = Up(64, mode)
        self.up2 = Up(64, mode)
        self.up3 = Up(64, mode)
        self.up4 = Up(64, 64, mode)
        self.outc = OutConv(64, n_classes)

        self.relu_g = nn.PReLU()
        self.gsig = nn.Sigmoid()

    def forward(self, x, g ):
        x1 = self.sfe(x)
        og1 = self.relu_g(self.sfeg(1-g))
        wg1 = self.gsig(og1)
        x1 = x1 * wg1 + x1
        x2 = self.down1(x1)
        
        og2 = self.dg2(og1)
        wg2 = self.gsig(og2)
        x2 = x2 * wg2 + x2
        x3 = self.down2(x2)

        og3 = self.dg3(og2)
        wg3 = self.gsig(og3)
        x3 = x3 * wg3 + x3
        x4 = self.down3(x3)

        og4 = self.dg4(og3)
        wg4 = self.gsig(og4)
        x4 = x4 * wg4 + x4
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return [logits]