# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import math
from modules import CARB,upsample_block
from models.modelG51 import MDDM

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.mddm_path1 = MDDM()
        self.mddm_path2 = MDDM()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, g):
        pim = self.mddm_path1(x)
        pg = self.mddm_path2(g)
        clean = pim/pg
        return [clean]

