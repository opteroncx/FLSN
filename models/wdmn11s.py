# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import math
from models.modules import upsample_block,Nonlocal_CA,WavePool,WaveUnpool,CARB,CARBH
from models import dncnn

class SKConv(nn.Module):
    def __init__(self, features=64, WH=32, M=2, G=8, r=2, stride=1 ,L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3+i*2, stride=stride, padding=1+i, groups=G),
                # nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

class SKUnit(nn.Module):
    def __init__(self, in_features=64, out_features=64, WH=32, M=2, G=8, r=2, mid_features=None, stride=1, L=32):
        """ Constructor  SKUnit(512, 512, 32, 2, 8, 2)
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features/2)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1),
            # nn.BatchNorm2d(mid_features),
            SKConv(mid_features, WH, M, G, r, stride=stride, L=L),
            # nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1, stride=1),
            # nn.BatchNorm2d(out_features)
        )
        if in_features == out_features: # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                # nn.BatchNorm2d(out_features)
            )
    
    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)

class WaveBlock(nn.Module):
    def __init__(self):
        super(WaveBlock, self).__init__()
        self.pool = WavePool(64)
        self.unpool = WaveUnpool(64,option_unpool='sum')
        self.block1 = nn.Sequential(SKUnit(64),ScaleLayer(init_value=1.0))
        self.block2 = nn.Sequential(SKUnit(64),ScaleLayer(init_value=1.0))
        self.block3 = nn.Sequential(SKUnit(64),ScaleLayer(init_value=1.0))
        self.block4 = nn.Sequential(SKUnit(64),ScaleLayer(init_value=1.0))
    
    def forward(self, x):
        ll, lh, hl, hh = self.pool(x)
        ll = self.block1(ll)
        lh = self.block2(lh)
        hl = self.block3(hl)
        hh = self.block4(hh)
        out = self.unpool(ll, lh, hl, hh)
        out = out + x
        return out

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

class Down2(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            SKUnit(64)
        )
    def forward(self, x):
        return self.maxpool_conv(x)


class Branch1(nn.Module):
    def __init__(self):
        super(Branch1, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_F11 = SKUnit(64)
        self.convt_F12 = SKUnit(64)
        self.convt_F13 = SKUnit(64)
        #-------------
        self.u1 = upsample_block(64,256)
        self.wb = WaveBlock()
        # self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        out = self.wb(out)
        convt_F11 = self.convt_F11(out)
        convt_F12 = self.convt_F12(convt_F11)
        convt_F13 = self.convt_F13(convt_F12)
        combine = out + convt_F13
        # clean = self.convt_shape1(up)
        return combine

class Branch2(nn.Module):
    def __init__(self):
        super(Branch2, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_F11 = SKUnit(64)
        self.convt_F12 = SKUnit(64)
        self.convt_F13 = SKUnit(64)
        #-------------
        self.u1 = upsample_block(64,256)
        self.wb = WaveBlock()
        # self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        out = self.wb(out)
        convt_F11 = self.convt_F11(out)
        convt_F12 = self.convt_F12(convt_F11)
        convt_F13 = self.convt_F13(convt_F12)
        combine = out + convt_F13
        up = self.u1(combine)
        return up

class Branch3(nn.Module):
    def __init__(self):
        super(Branch3, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_F11 = SKUnit(64)
        self.convt_F12 = SKUnit(64)
        self.convt_F13 = SKUnit(64)

        self.u1 = upsample_block(64,256)
        self.u2 = upsample_block(64,256)
        self.wb = WaveBlock()
        # self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.non_local = Nonlocal_CA(in_feat=64, inter_feat=64//8, reduction=8,sub_sample=False, bn_layer=False)


    def forward(self, x):
        out = self.relu(self.conv_input(x))
        out = self.wb(out)
        convt_F11 = self.convt_F11(out)             
        convt_F12 = self.convt_F12(convt_F11)      
        convt_F13 = self.convt_F13(convt_F12)
        # convt_F15 = self.non_local(convt_F15)
        combine = out + convt_F13
        up = self.u1(combine)
        up = self.u2(up)
        return up

class Branch4(nn.Module):
    def __init__(self):
        super(Branch4, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_F11 = SKUnit(64)
        self.convt_F12 = SKUnit(64)
        self.convt_F13 = SKUnit(64)

        self.u1 = upsample_block(64,256)
        self.u2 = upsample_block(64,256)
        self.u3 = upsample_block(64,256)
        self.wb = WaveBlock()
        # self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.non_local = Nonlocal_CA(in_feat=64, inter_feat=64//8, reduction=8,sub_sample=False, bn_layer=False)


    def forward(self, x):
        out = self.relu(self.conv_input(x))
        out = self.wb(out)
        convt_F11 = self.convt_F11(out)               
        convt_F12 = self.convt_F12(convt_F11)
        convt_F13 = self.convt_F13(convt_F12)
        # convt_F14 = self.non_local(convt_F14)
        combine = out + convt_F13
        up = self.u1(combine)
        up = self.u2(up)
        up = self.u3(up)
        return up

class OutConv(nn.Module):
    def __init__(self,ichannels=64):
        super(OutConv, self).__init__()
        self.conv_tail = nn.Conv2d(in_channels=ichannels, out_channels=ichannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.conv_out = nn.Conv2d(ichannels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, f):
        tail = self.relu(self.conv_tail(f))
        out = self.conv_out(tail)
        return out

class Net(nn.Module):
    def __init__(self, n_channels=1):
        super(Net, self).__init__()
        self.LE_head = dncnn.Net(channels=n_channels)
        self.inc = DoubleConv(2*n_channels, 64)
        self.down2_1 = Down2()
        self.down2_2 = Down2()
        self.down2_3 = Down2()
        # Branches
        # self.branch1 = Branch1()
        self.branch2 = Branch2()
        self.branch3 = Branch3()
        self.branch4 = Branch4()
        # self.scale1 = ScaleLayer()
        self.scale2 = ScaleLayer()
        self.scale3 = ScaleLayer()
        self.scale4 = ScaleLayer()
        self.out = OutConv(64)

    def forward(self, x):
        x0 = self.LE_head(x)
        x1 = self.inc(torch.cat([x0,x],dim=1))
        # b1 = self.branch1(x1)
        # b1 = self.scale1(b1)

        feat_down2 = self.down2_1(x1)
        b2 = self.branch2(feat_down2)
        b2 = self.scale2(b2)

        feat_down4 = self.down2_2(feat_down2)
        b3 = self.branch3(feat_down4)
        b3 = self.scale3(b3)        

        feat_down8 = self.down2_3(feat_down4)
        b4 = self.branch4(feat_down8)
        b4 = self.scale4(b4)           
        #--------- 
        fuse = b2 + b3 + b4
        clean = self.out(fuse)
        clean = clean + x
        return clean

class ScaleLayer(nn.Module):
   def __init__(self, init_value=1.0):
       super(ScaleLayer,self).__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))

   def forward(self, x):
       return x * self.scale

