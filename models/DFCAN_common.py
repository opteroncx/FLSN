import numpy as np
# import tensorflow as tf
import torch
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x)

def gelu(x):
    gelu = GELU()
    return gelu(x)

#torch默认的数据格式应该是BCHW，tensorflow的是BHWC
# def fft2d(input, gamma=0.1):
#     temp = K.permute_dimensions(input, (0, 3, 1, 2))
#     fft = tf.fft2d(tf.complex(temp, tf.zeros_like(temp)))
#     absfft = tf.pow(tf.abs(fft)+1e-8, gamma)
#     output = K.permute_dimensions(absfft, (0, 2, 3, 1))
#     return output


# def fftshift2d(input, size_psc=128):
#     # bs, h, w, ch = input.get_shape().as_list()
#     bs, h, w, ch = input.shape()
#     fs11 = input[:, -h // 2:h, -w // 2:w, :]
#     fs12 = input[:, -h // 2:h, 0:w // 2, :]
#     fs21 = input[:, 0:h // 2, -w // 2:w, :]
#     fs22 = input[:, 0:h // 2, 0:w // 2, :]
#     output = torch.cat([torch.cat([fs11, fs21], 2), torch.cat([fs12, fs22], 2)], 3)  #torch默认的数据格式应该是BCHW，tensorflow的是BHWC
#     output = torch.nn.functional.interpolate(output, (size_psc, size_psc), mode='bilinear')
#     # output = tf.image.resize_images(output, (size_psc, size_psc), 0)
#     return output

def fft2d(x):
    return torch.fft.fft2(x)

def fftshift2d(x,target_size=128):
    return torch.fft.fftshift(x)

class upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsample_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.shuffler(self.gelu(self.conv(x)))

# def pixel_shiffle(layer_in, scale):
#     return tf.depth_to_space(layer_in, block_size=scale)


class Global_average_pooling2d(nn.Module):
    def __init__(self):
        super(Global_average_pooling2d, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(x)

def global_average_pooling2d(x):
    # return torch.mean(layer_in, axis=(1, 2), keepdims=True)
    pool = nn.AdaptiveAvgPool2d(1)
    return pool(x)
