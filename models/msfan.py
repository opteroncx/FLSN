# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import math
from models.modules import upsample_block,Nonlocal_CA,CAT,SOCARB,CARB,FRM,Space_attention,UpBlock,DownBlock

def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)

class RK3(nn.Module):
    def __init__(self, n_feats=64, kernel_size=3,bias=True, act=nn.PReLU(1, 0.25), res_scale=1):

        super(RK3, self).__init__()

        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv3 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu1 = nn.PReLU(n_feats, 0.25)
        self.relu2 = nn.PReLU(n_feats, 0.25)
        self.relu3 = nn.PReLU(n_feats, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.scale2 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)
        self.scale3 = nn.Parameter(torch.FloatTensor([-1.0]), requires_grad=True)
        self.scale4 = nn.Parameter(torch.FloatTensor([4.0]), requires_grad=True)
        self.scale5 = nn.Parameter(torch.FloatTensor([1/6]), requires_grad=True)

    def forward(self, x):
        
        yn = x
        k1 = self.relu1(x)
        k1 = self.conv1(k1)
        yn_1 = k1*self.scale1 + yn
        k2 = self.relu2(yn_1)
        k2 = self.conv2(k2)
        yn_2 = yn + self.scale2*k2
        yn_2 = yn_2 + k1*self.scale3
        k3 = self.relu3(yn_2)
        k3 = self.conv3(k3)
        yn_3 = k3 + k2*self.scale4 + k1
        yn_3 = yn_3*self.scale5
        out = yn_3 + yn
        return out

class Down2(nn.Module):
    def __init__(self,c_in,c_out):
        super(Down2, self).__init__()
        
        self.conv_input = nn.Conv2d(in_channels=c_in, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_R1 = nn.Conv2d(in_channels=32, out_channels=c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.block = CARB(64)
        self.down = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()            

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
        # self.convt_F04 = CARB(64)
        # self.convt_F05 = CARB(64)

        self.convt_F11 = CARB(64)
        self.convt_F12 = CARB(64)
        self.convt_F13 = CARB(64)
        self.convt_F14 = CARB(64)
        # self.convt_F15 = CARB(64)
        # self.convt_F16 = CARB(64)
        # self.convt_F17 = CARB(64)
        # self.convt_F18 = RK3()
        # self.convt_F19 = RK3()
        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.cats = CAT(128)
        self.SA1 = Space_attention(64,64,1,1,0,1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, b):
        out = self.relu(self.conv_input(x))
        convt_F01 = self.convt_F01(out)
        convt_F02 = self.convt_F02(convt_F01)
        shallow_ft = self.convt_F03(convt_F02)
        # convt_F04 = self.convt_F04(convt_F03)
        # shallow_ft = self.convt_F05(convt_F04)

        fu = torch.cat((shallow_ft,b),1)
        fu = self.cats(fu)
        # fu = self.SA1(fu)
        cf1 = self.convt_F11(fu)
        cf2 = self.convt_F12(cf1)
        cf3 = self.convt_F13(cf2)
        cf4 = self.convt_F14(cf3)
        # cf5 = self.convt_F15(cf4)
        # cf6 = self.convt_F16(cf5)
        # cf7 = self.convt_F17(cf6)
        # cf8 = self.convt_F18(cf7)
        # cf9 = self.convt_F19(cf8)
        # clean = self.conv_input2(cf6)
        clean = cf4
        return clean

class Branch2(nn.Module):
    def __init__(self):
        super(Branch2, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_input2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_F01 = CARB(64)
        self.convt_F02 = CARB(64)
        self.convt_F03 = CARB(64)
        # self.convt_F04 = CARB(64)
        # self.convt_F05 = CARB(64)

        self.convt_F11 = CARB(64)
        self.convt_F12 = CARB(64)
        self.convt_F13 = CARB(64)
        self.convt_F14 = CARB(64)
        # self.convt_F15 = CARB(64)
        # self.convt_F16 = CARB(64)
        # self.convt_F17 = CARB(64)
        # self.convt_F18 = RK3()
        # self.convt_F19 = RK3()
        #-------------
        self.cats = CAT(128)
        self.u1 = upsample_block(64,256)
        # self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        # self.SA1 = Space_attention(64,64,1,1,0,1)
        # self.SA2 = Space_attention(64,64,1,1,0,1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, b):
        out = self.relu(self.conv_input(x))
        convt_F01 = self.convt_F01(out)
        convt_F02 = self.convt_F02(convt_F01)
        shallow_ft = self.convt_F03(convt_F02)
        # convt_F04 = self.convt_F04(convt_F03)
        # shallow_ft = self.convt_F05(convt_F04)

        fu = torch.cat((shallow_ft,b),1)
        fu = self.cats(fu)
        # fu = self.SA1(fu)
        convt_F11 = self.convt_F11(fu)
        convt_F12 = self.convt_F12(convt_F11)
        convt_F13 = self.convt_F13(convt_F12)
        convt_F14 = self.convt_F14(convt_F13)
        # convt_F15 = self.convt_F15(convt_F14)
        # convt_F16 = self.convt_F16(convt_F15)
        # convt_F17 = self.convt_F17(convt_F16)
        # convt_F18 = self.convt_F18(convt_F17)
        # convt_F19 = self.convt_F19(convt_F18)
        #上采样
        combine = out + convt_F14
        # combine = self.SA2(combine)
        up = self.u1(combine)
        f = up
        # clean = self.convt_shape1(up)
        clean = up
        return clean,f

class Branch3(nn.Module):
    def __init__(self):
        super(Branch3, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_input2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_F11 = CARB(64)
        self.convt_F12 = CARB(64)
        self.convt_F13 = CARB(64)
        self.convt_F14 = CARB(64)
        # self.convt_F15 = CARB(64)
        # self.convt_F16 = CARB(64)
        # self.convt_F17 = CARB(64)
        # self.convt_F18 = RK3()
        # self.convt_F19 = RK3()
        # self.convt_F20 = RK3()
        self.u1 = upsample_block(64,256)
        self.u2 = upsample_block(64,256)
        self.cats = CAT(128)
        # self.SA1 = Space_attention(64,64,1,1,0,1)
        # self.SA2 = Space_attention(64,64,1,1,0,1)
        # self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        # self.non_local = Nonlocal_CA(in_feat=64, inter_feat=64//8, reduction=8,sub_sample=False, bn_layer=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        convt_F11 = self.convt_F11(out)            
        convt_F12 = self.convt_F12(convt_F11)  
        convt_F13 = self.convt_F13(convt_F12)
        convt_F14 = self.convt_F14(convt_F13)
        # convt_F15 = self.convt_F15(convt_F14)
        # convt_F16 = self.convt_F16(convt_F15)
        # convt_F17 = self.convt_F17(convt_F16)
        # convt_F18 = self.convt_F18(convt_F17)
        # convt_F19 = self.convt_F19(convt_F18)
        # convt_F20 = self.convt_F20(convt_F19)
        # convt_F14 = self.non_local(convt_F14)
        #上采样
        combine = out + convt_F14
        # combine = self.SA2(combine)
        up = self.u1(combine)
        f = up
        up = self.u2(up)
        # clean = self.convt_shape1(up)
        clean = up
        return clean,f

class To_clean_image(nn.Module):
    def __init__(self,ichannels=64):
        super(To_clean_image, self).__init__()
        self.se = FRM(ichannels)
        self.gff = nn.Conv2d(in_channels=ichannels, out_channels=ichannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.conv_tail = nn.Conv2d(in_channels=ichannels, out_channels=ichannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relut = nn.PReLU()
        self.conv_out = nn.Conv2d(ichannels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, resize1):
        f = resize1
        # gff = self.relu(self.gff(concat))
        # se = self.se(gff)+concat
        tail = self.relut(self.conv_tail(f))
        out = self.conv_out(tail)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 下采样
        self.down2_1 = Down2(1,64)
        self.down2_2 = Down2(64,64)
        # self.down2_3a = Down2(64,64)
        # Branches
        self.branch1 = Branch1()
        self.branch2 = Branch2()
        self.branch3 = Branch3()
        # self.branch4 = Branch4()
        # self.SA2 = Space_attention(64,64,1,1,0,1)
        # self.SA3 = Space_attention(64,64,1,1,0,1)

        self.to_clean1 = To_clean_image()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # out = self.relu(self.conv_input(x))
        feat_down2 = self.down2_1(x)
        # print('3') 32x32
        feat_down3 = self.down2_2(feat_down2)
        # feat_down4 = self.down2_3(feat_down3)
        #----------------------------------------------------------------
        # i4,b4 = self.branch4(feat_down4)              
        #---------
        i3,b3 = self.branch3(feat_down3)  
        #---------         
        # feat_down2 = self.SA2(feat_down2)
        i2,b2 = self.branch2(feat_down2,b3)
        #---------
        i1 = self.branch1(x,b2)
        #---------
        clean = self.to_clean1(i1)
        # clean = self.convt_shape1(combine)
        return clean

class ScaleLayer(nn.Module):

   def __init__(self, init_value=1.0):
       super(ScaleLayer,self).__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))

   def forward(self, x):
    #    print(self.scale)
       return x * self.scale

