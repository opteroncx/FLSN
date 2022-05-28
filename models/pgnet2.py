# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.modules import upsample_block,CARB,CAT,Space_attention

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

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

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
    def __init__(self, in_channels, out_channels, up_mode='bicubic'):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
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

class LLDNN(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(LLDNN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        mode = 'bicubic'

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 64)
        self.down2 = Down(64, 64)
        self.down3 = Down(64, 64)

        self.down4 = Down(64, 64)
        self.up1 = Up(64, mode)
        self.up2 = Up(64, mode)
        self.up3 = Up(64, mode)
        self.up4 = Up(64, mode)
        self.out = OutConv(64,1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x

class Down2(nn.Module):
    def __init__(self,c_in,c_out):
        super(Down2, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.block = CARB(64)
        self.down = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)       

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        out = self.down(out)
        LR_2x = self.convt_R1(out)
        LR_2x = self.block(LR_2x)
        return LR_2x

class Branch1(nn.Module):
    def __init__(self):
        super(Branch1, self).__init__()
        self.convt_F01 = CARB(64)
        self.convt_F02 = CARB(64)
        self.convt_F03 = CARB(64)

        self.convt_F11 = CARB(64)
        self.convt_F12 = CARB(64)
        self.convt_F13 = CARB(64)
        self.convt_F14 = CARB(64)
        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.cats = CAT(128)

    def forward(self, x, b):
        out = self.relu(self.conv_input(x))
        convt_F01 = self.convt_F01(out)
        convt_F02 = self.convt_F02(convt_F01)
        shallow_ft = self.convt_F03(convt_F02)
        fu = torch.cat((shallow_ft,b),1)
        fu = self.cats(fu)
        cf1 = self.convt_F11(fu)
        cf2 = self.convt_F12(cf1)
        cf3 = self.convt_F13(cf2)
        cf4 = self.convt_F14(cf3)
        clean = cf4
        return clean

class Branch2(nn.Module):
    def __init__(self):
        super(Branch2, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_F01 = CARB(64)
        self.convt_F02 = CARB(64)
        self.convt_F03 = CARB(64)

        self.convt_F11 = CARB(64)
        self.convt_F12 = CARB(64)
        self.convt_F13 = CARB(64)
        self.convt_F14 = CARB(64)

        self.cats = CAT(128)
        self.u1 = upsample_block(64,256)

    def forward(self, x, b):
        out = self.relu(self.conv_input(x))
        convt_F01 = self.convt_F01(out)
        convt_F02 = self.convt_F02(convt_F01)
        shallow_ft = self.convt_F03(convt_F02)
        fu = torch.cat((shallow_ft,b),1)
        fu = self.cats(fu)
        convt_F11 = self.convt_F11(fu)
        convt_F12 = self.convt_F12(convt_F11)
        convt_F13 = self.convt_F13(convt_F12)
        convt_F14 = self.convt_F14(convt_F13)
        combine = out + convt_F14
        up = self.u1(combine)
        return up

class Branch3(nn.Module):
    def __init__(self):
        super(Branch3, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_F11 = CARB(64)
        self.convt_F12 = CARB(64)
        self.convt_F13 = CARB(64)
        self.convt_F14 = CARB(64)
        self.u1 = upsample_block(64,256)
        self.u2 = upsample_block(64,256)
        self.cats = CAT(128)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        convt_F11 = self.convt_F11(out)            
        convt_F12 = self.convt_F12(convt_F11)  
        convt_F13 = self.convt_F13(convt_F12)
        convt_F14 = self.convt_F14(convt_F13)
        combine = out + convt_F14
        up = self.u1(combine)
        return up

class To_clean_image(nn.Module):
    def __init__(self,ichannels=64):
        super(To_clean_image, self).__init__()
        self.conv_tail = nn.Conv2d(in_channels=ichannels, out_channels=ichannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.conv_out = nn.Conv2d(ichannels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, f):
        tail = self.relu(self.conv_tail(f))
        out = self.conv_out(tail)
        return out

class SRDMN(nn.Module):
    def __init__(self):
        super(SRDMN, self).__init__()
        # 下采样
        self.down2_1 = Down2(1,64)
        self.down2_2 = Down2(64,64)
        # Branches
        self.branch1 = Branch1()
        self.branch2 = Branch2()
        self.branch3 = Branch3()
        self.to_clean = To_clean_image()

    def forward(self, x):
        feat_down2 = self.down2_1(x)
        # print('3') 32x32
        feat_down3 = self.down2_2(feat_down2)
        b3 = self.branch3(feat_down3)  
        b2 = self.branch2(feat_down2,b3)
        i1 = self.branch1(x,b2)
        clean = self.to_clean(i1)
        return clean

class ScaleLayer(nn.Module):
   def __init__(self, init_value=1.0):
       super(ScaleLayer,self).__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))

   def forward(self, x):
       return x * self.scale


# Network definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.LLDN = LLDNN()
        self.SRDM = SRDMN()
        self.scale1 = ScaleLayer(0.5)
        self.scale2 = ScaleLayer(0.5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, LELR):
        G_1 = self.LLDN(LELR)
        G_2 = self.SRDM(LELR)
        G_HEHR = self.scale1(G_1)+self.scale2(G_2)
        return G_HEHR