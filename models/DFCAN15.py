import torch
import torch.nn as nn
import torch.fft as fft
import math

# def fft2d(x):
#     return torch.fft.fft2(x)

# def fftshift2d(x,target_size=128):
#     return torch.fft.fftshift(x)

#torch默认的数据格式应该是BCHW，tensorflow的是BHWC
def fft2d(x, gamma=0.1):
    # temp = K.permute_dimensions(input, (0, 3, 1, 2))  BHWC -> BCHW
    temp = x   # pytorch keep dim orders
    # fft = tf.fft2d(tf.complex(temp, tf.zeros_like(temp)))
    fft = torch.fft.fft2(temp)
    absfft = torch.pow(torch.abs(fft)+1e-8, gamma)
    # output = K.permute_dimensions(absfft, (0, 2, 3, 1)) BCHW -> BHWC
    output = absfft   # pytorch keep dim orders
    return output


def fftshift2d(input, size_psc=128):
    # bs, h, w, ch = input.get_shape().as_list()
    bs, ch, h, w = input.shape
    fs11 = input[:,:, -h // 2:h, -w // 2:w]
    fs12 = input[:,:, -h // 2:h, 0:w // 2]
    fs21 = input[:,:, 0:h // 2, -w // 2:w]
    fs22 = input[:,:, 0:h // 2, 0:w // 2]
    output = torch.cat([torch.cat([fs11, fs21], 2), torch.cat([fs12, fs22], 2)], 3)  #torch默认的数据格式应该是BCHW，tensorflow的是BHWC
    # output = torch.nn.functional.interpolate(output, (size_psc, size_psc), mode='bilinear')
    # output = tf.image.resize_images(output, (size_psc, size_psc), 0)
    # print(output.shape)
    return output


class upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsample_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.shuffler(self.gelu(self.conv(x)))

class FCALayer(nn.Module):
    def __init__(self, channel, reduction=16, size_psc=128):
        super(FCALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.cd = nn.Conv2d(64, 4, 1, padding=0, bias=True)
        self.relu = nn.ReLU()
        self.cu = nn.Conv2d(4, 64, 1, padding=0, bias=True)
        self.sig = nn.Sigmoid()
        self.size_psc = size_psc

    def forward(self, x):
        absfft1 = fft2d(x)
        absfft1 = fftshift2d(absfft1,self.size_psc)
        y = self.avg_pool(absfft1)
        y = self.cd(y)
        y = self.relu(y)
        y = self.cu(y)
        y = self.sig(y)
        return x * y

# class FCALayer1(nn.Module):
#     def __init__(self,channel, reduction=16, size_psc=128):
#         super(FCALayer1, self).__init__()
#         self.conv = nn.Conv2d(channel,channel,kernel_size=3,padding=1)
#         self.relu = nn.ReLU()
#         # global average pooling: feature --> point
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#                 nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#                 nn.Sigmoid()
#         )
#         self.size_psc = size_psc

#     def forward(self, x):
#         absfft1 = fft2d(x)
#         absfft1 = fftshift2d(absfft1,self.size_psc)
#         # absfft1 = x
#         absfft2 = self.relu(self.conv(absfft1))
#         y = self.avg_pool(absfft2)
#         y = self.conv_du(y)
#         return y*x

# def FCALayer(input, channel, reduction=16, size_psc=128):
#     # absfft1 = Lambda(fft2d, arguments={'gamma': 0.8})(input)
#     absfft1 = fft2d(input)
#     # absfft1 = Lambda(fftshift2d, arguments={'size_psc': size_psc})(absfft1)
#     absfft1 = fftshift2d(absfft1)
#     absfft2 = Conv2D(channel, kernel_size=3, activation='relu', padding='same')(absfft1)
#     W = Lambda(global_average_pooling2d)(absfft2)
#     W = Conv2D(channel // reduction, kernel_size=1, activation='relu', padding='same')(W)
#     W = Conv2D(channel, kernel_size=1, activation='sigmoid', padding='same')(W)
#     mul = multiply([input, W])
#     return mul


class FCAB(nn.Module):
    def __init__(self,channel, size_psc=128):
        super(FCAB, self).__init__()
        self.conv1 = nn.Conv2d(channel,channel,kernel_size=3,padding=1)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(channel,channel,kernel_size=3,padding=1)
        self.gelu2 = nn.GELU()
        self.attention = FCALayer(channel,size_psc)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.gelu1(x1)
        x2 = self.conv2(x1)
        x2 = self.gelu2(x2)
        att = self.attention(x2)
        return att+x

# def FCAB(input, channel, size_psc=128):
#     conv = Conv2D(channel, kernel_size=3, padding='same')(input)
#     conv = Lambda(gelu)(conv)
#     conv = Conv2D(channel, kernel_size=3, padding='same')(conv)
#     conv = Lambda(gelu)(conv)
#     att = FCALayer(conv, channel, reduction=16, size_psc=size_psc)
#     output = add([att, input])
#     return output


class ResidualGroup(nn.Module):
    def __init__(self,channel, size_psc=128):
        super(ResidualGroup, self).__init__()
        n_RCAB = 4
        # layers = []
        # for _ in range(n_RCAB):
        #     layers.append(FCAB(channel,size_psc))
        # self.group = nn.Sequential(*layers)
        self.group = FCAB(channel,size_psc)

    def forward(self, x):
        return self.group(x)


# def ResidualGroup(input, channel, size_psc=128):
#     conv = input
#     n_RCAB = 4
#     for _ in range(n_RCAB):
#         conv = FCAB(conv, channel=channel, size_psc=size_psc)
#     conv = add([conv, input])
#     return conv

class DFCAN(nn.Module):
    def __init__(self,channel=64,chennel_out=1, size_psc=128, scale=1):
        super(DFCAN, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(15,channel,kernel_size=3,padding=1)
        self.gelu1 = nn.GELU()
        layers = []
        n_ResGroup = 4
        for _ in range(n_ResGroup):
            layers.append(ResidualGroup(channel,size_psc))
        self.conv2 = nn.Sequential(*layers)

        self.upsampled = upsample_block(channel,channel * (scale ** 2))    
        self.out = nn.Conv2d(channel,chennel_out,kernel_size=3,padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.gelu1(self.conv1(x))
        conv2 = self.conv2(conv1)
        if self.scale > 1:
            up = self.upsampled(conv2)
        else:
            up = conv2
        out = self.sig(self.out(up))
        return out

# def DFCAN(input_shape, scale=2, size_psc=128):
#     inputs = Input(input_shape)
#     conv = Conv2D(64, kernel_size=3, padding='same')(inputs)
#     conv = Lambda(gelu)(conv)
#     n_ResGroup = 4
#     for _ in range(n_ResGroup):
#         conv = ResidualGroup(conv, 64, size_psc)
#     conv = Conv2D(64 * (scale ** 2), kernel_size=3, padding='same')(conv)
#     conv = Lambda(gelu)(conv)
#     upsampled = Lambda(pixel_shiffle, arguments={'scale': scale})(conv)
#     conv = Conv2D(1, kernel_size=3, padding='same')(upsampled)
#     output = Activation('sigmoid')(conv)
#     model = Model(inputs=inputs, outputs=output)
#     return model