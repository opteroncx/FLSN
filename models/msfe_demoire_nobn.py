# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import math
from modules import MLB,CARB,upsample_block
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(ResBlock, self).__init__()
        cin = inchannels
        cout = outchannels
        self.trans = nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=cout, out_channels=cout, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(cout)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=cout, out_channels=cout, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(cout)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()            
    def forward(self, x):
        x = self.trans(x)
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        return out + x

class Down0(nn.Module):
    def __init__(self):
        super(Down0, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2, bias=False)
        self.down = nn.MaxPool2d(2)
        self.rb = ResBlock(16,32)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()            

    def forward(self, x):
        out = self.relu(self.conv(x))
        out = self.down(out)
        out = self.rb(out)
        return out

class Down1(nn.Module):
    def __init__(self):
        super(Down1, self).__init__()
        self.down = nn.MaxPool2d(2)
        self.rb = ResBlock(32,64)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()            

    def forward(self, x):
        out = self.down(x)
        LR_2x = self.rb(out)
        return LR_2x

class Down2(nn.Module):
    def __init__(self):
        super(Down2, self).__init__()
        self.down = nn.MaxPool2d(2)
        self.rb = ResBlock(64,128)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()            

    def forward(self, x):
        out = self.down(x)
        LR_2x = self.rb(out)
        return LR_2x

class Down3(nn.Module):
    def __init__(self):
        super(Down3, self).__init__()
        self.down = nn.MaxPool2d(2)
        self.rb = ResBlock(128,256)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()            

    def forward(self, x):
        out = self.down(x)
        LR_2x = self.rb(out)
        return LR_2x

class Dec1(nn.Module):
    def __init__(self):
        super(Dec1, self).__init__()
        self.rb = ResBlock(256,256)
        self.conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.u1 = nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=2,stride=2,padding=0)
        self.u2 = nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=2,stride=2,padding=0)
        self.u3 = nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=2,stride=2,padding=0)
        self.u4 = nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=2,stride=2,padding=0)
        self.relu1 = nn.ReLU()    
        self.relu2 = nn.ReLU()     
        self.relu3 = nn.ReLU() 
        self.relu4 = nn.ReLU() 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.rb(x)
        f0 = self.relu1(self.u1(out))
        out = self.relu2(self.u2(f0))
        out = self.relu3(self.u3(out))
        out = self.relu4(self.u4(out))
        clean = self.conv(out)
        return clean

class Dec2(nn.Module):
    def __init__(self):
        super(Dec2, self).__init__()
        self.rb = ResBlock(128,128)
        self.conv = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.u1 = nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=2,stride=2,padding=0)
        self.u2 = nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=2,stride=2,padding=0)
        self.u3 = nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=2,stride=2,padding=0)
        self.relu1 = nn.ReLU()    
        self.relu2 = nn.ReLU()     
        self.relu3 = nn.ReLU()
        self.m1 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x,f0):
        f0 = self.m1(f0)
        uf0 = F.upsample(f0,scale_factor=2,mode='bilinear')
        mix = x*uf0
        out = self.rb(mix)
        f0 = self.relu1(self.u1(out))
        out = self.relu2(self.u2(f0))
        out = self.relu3(self.u3(out))
        clean = self.conv(out)
        return clean

class Dec3(nn.Module):
    def __init__(self):
        super(Dec3, self).__init__()
        self.rb = ResBlock(64,64)
        self.conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.u1 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=2,stride=2,padding=0)
        self.u2 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=2,stride=2,padding=0)
        self.relu1 = nn.ReLU()    
        self.relu2 = nn.ReLU() 
        self.m1 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.m2 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x,f0,f1):
        f0 = self.m1(f0)
        uf0 = F.upsample(f0,scale_factor=2,mode='bilinear')
        f1 = self.m2(f1)
        uf1 = F.upsample(f1,scale_factor=4,mode='bilinear')
        mix = x*uf0*uf1
        out = self.rb(mix)
        f0 = self.relu1(self.u1(out))
        out = self.relu2(self.u2(f0))
        clean = self.conv(out)
        return clean

class Dec4(nn.Module):
    def __init__(self):
        super(Dec4, self).__init__()
        self.rb = ResBlock(32,32)
        self.conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.u1 = nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=2,stride=2,padding=0)
        self.relu1 = nn.ReLU()
        self.m1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.m2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)  
        self.m3 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False) 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x,f0,f1,f2):
        f0 = self.m1(f0)
        uf0 = F.upsample(f0,scale_factor=2,mode='bilinear')
        f1 = self.m2(f1)
        uf1 = F.upsample(f1,scale_factor=4,mode='bilinear')
        f2 = self.m3(f2)
        uf2 = F.upsample(f2,scale_factor=8,mode='bilinear')
        mix = x*uf0*uf1*uf2
        out = self.rb(mix)
        out = self.relu1(self.u1(out))
        clean = self.conv(out)
        return clean

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.relu = nn.PReLU()
        # 下采样
        self.down0 = Down0()
        self.down1 = Down1()
        self.down2 = Down2()
        self.down3 = Down3()
        # Branches
        self.dec1 = Dec1()
        self.dec2 = Dec2()
        self.dec3 = Dec3()
        self.dec4 = Dec4()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        #---------
        feat_down1 = self.down0(x)
        feat_down2 = self.down1(feat_down1)
        feat_down3 = self.down2(feat_down2)
        feat_down4 = self.down3(feat_down3)

        b1 = self.dec1(feat_down4)
        b2 = self.dec2(feat_down3,feat_down4)
        b3 = self.dec3(feat_down2,feat_down3,feat_down4)
        b4 = self.dec4(feat_down1,feat_down2,feat_down3,feat_down4)
        clean = b1 + b2 + b3 + b4

        return [clean]

