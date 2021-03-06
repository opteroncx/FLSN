# coding:utf8
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import time
import numpy as np


def print_model_parm_nums(model):
    # model = models.alexnet()
    # from model5_adain import Net
    # model = Net()
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))


def print_model_parm_flops():

    # prods = {}
    # def save_prods(self, input, output):
    # print 'flops:{}'.format(self.__class__.__name__)
    # print 'input:{}'.format(input)
    # print '_dim:{}'.format(input[0].dim())
    # print 'input_shape:{}'.format(np.prod(input[0].shape))
    # grads.append(np.prod(input[0].shape))

    prods = {}

    def save_hook(name):
        def hook_per(self, input, output):
            # print 'flops:{}'.format(self.__class__.__name__)
            # print 'input:{}'.format(input)
            # print '_dim:{}'.format(input[0].dim())
            # print 'input_shape:{}'.format(np.prod(input[0].shape))
            # prods.append(np.prod(input[0].shape))
            prods[name] = np.prod(input[0].shape)
            # prods.append(np.prod(input[0].shape))
        return hook_per

    list_1 = []

    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2 = {}

    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    multiply_adds = False
    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * \
            (self.in_channels / self.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)

    # resnet = models.alexnet()
    from models.scUnet import Net as Net
    # TensorRT
    # from torch2trt import torch2trt
    resnet = Net()
    foo(resnet)
    cudnn.benchmark= True
    # input = torch.rand(1,256,256).unsqueeze(0)
    # x = torch.ones((4, 1, 256, 256)).cuda()
    x = torch.ones((1, 1, 256, 256)).cuda()
    # x = torch.ones((1, 1, 128, 128)).cuda().half()
    # model_trt = torch2trt(resnet, [x])
    resnet = resnet.cuda()
    # input = input.cuda()
    time_list = []
    with torch.no_grad():
        iters = 1000
        for i in range(iters):
            start_time = time.time()
            out = resnet(x)
            elapsed = time.time() - start_time
            time_list.append(elapsed)
    print("Elapsed time: {}".format(np.amin(time_list)))

    total_flops = (sum(list_conv) + sum(list_linear) +
                   sum(list_bn) + sum(list_relu) + sum(list_pooling))

    print('  + Number of FLOPs: %.2fG' % (total_flops / (1e9*iters)))
    print_model_parm_nums(resnet)
    # print list_bn

    # print 'prods:{}'.format(prods)
    # print 'list_1:{}'.format(list_1)
    # print 'list_2:{}'.format(list_2)
    # print 'list_final:{}'.format(list_final)


if __name__ == '__main__':
    print_model_parm_flops()
    # print_model_parm_nums()
