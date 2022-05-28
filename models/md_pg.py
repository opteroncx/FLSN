import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import CARB

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

class ReLight(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.block1 = CARB(channels)
        self.block2 = CARB(channels)
        self.block3 = CARB(channels)
    
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        out = x3+x
        return out

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode='bicubic'):
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
        print(diffX)#检查一下这两个是不是0
        print(diffY)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)+x

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
        self.rl = ReLight(64)

        self.down1 = Down(64, 64)
        self.down2 = Down(64, 64)
        self.down3 = Down(64, 64)
        self.down4 = Down(64, 64)

        self.up1 = Up(64, mode)
        self.up2 = Up(64, mode)
        self.up3 = Up(64, mode)
        self.up4 = Up(64, 64, mode)
        self.outc = OutConv(64, n_classes)

        self.relu_g = nn.PReLU()
        self.gsig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.sfe(x)
        x1 = self.rl(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits