from absl import flags, app
from absl.flags import FLAGS
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler
import random
from torch.utils.data import Dataset, DataLoader
import torch
import time, statistics
from randpointslib.getData import getData
from matplotlib import pyplot
import pandas as pd

flags.DEFINE_float('learning_rate', 
                    0.1, 
                    'set the learning rate {float}. default: 0.1')

flags.DEFINE_integer('batch_size', 
                    16, 
                    'set the batch size {integer}. default: 8')

flags.DEFINE_integer('n_epochs', 
                    20, 
                    'set the number of epochs {integer}. default: 20')

flags.DEFINE_float('schedFactor', 
                    0.1, 
                    'set the factor of the scheduler {float}. default: 0.1')

flags.DEFINE_integer('schedPatience', 
                    10, 
                    'set the patience of the scheduler {integer}. default: 10')

flags.DEFINE_float('weight_decay', 
                    0., 
                    'set the weight decay {float}. default: 0.')

flags.DEFINE_boolean('printShape', 
                    False, 
                    'set if the shape of the layers are being printed {boolean}. default: False')

flags.DEFINE_list('o_channels', 
                    [4, 4, 4, 4], 
                    'set a list of channel size for each HIDDEN conv layer {integer}. The last layer is hardcoded to 1 (same as input - Regression). default: default: 4,4,4')

flags.DEFINE_list('kernel_sizes', 
                    [3, 3, 3, 3], 
                    'set a list of kernel size for each conv layer {integer}. One number represents the H and the W of the Kernel size. It uses rectangular kernels. default: 3,3,3,3')

flags.DEFINE_list('strides', 
                    [1, 1, 1, 1], 
                    'set a list of strides for each conv layer {integer}. default: 1,1,1,1')

flags.DEFINE_list('paddings', 
                    [1, 1, 1, 1], 
                    'set a list of padding size for each conv layer {integer}. default: 1,1,1,1')

flags.DEFINE_list('dilations', 
                    [1, 1, 1, 1], 
                    'set a list of dilation size for each conv layer {integer}. default: 1,1,1,1')

flags.DEFINE_string('load_model',
                    '',
                    'Set a path to a model. default: \'\'')

# Layer with integrated Convolutional, BatchNormalization.
class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, name="1", printForward=False):
        super(ConvBn2d, self).__init__()
        self.printShape = printForward
        self.name = "conv_bn_2d_"+str(name)
        self.conv = nn.Conv2d(in_channels = in_channels, 
                              out_channels = out_channels, 
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = padding,
                              dilation = dilation)
        self.bn = nn.BatchNorm2d(num_features = out_channels)

    def forward(self, x):
        out = self.conv(x)
        if self.printShape:
            print(f"{self.name} : {out.shape}")
        out = self.bn(out)
        if self.printShape:
            print(f"{self.name} : {out.shape}")
        
        return out

# TODO: fix issues and try to integrate to the DenooiseNet
class ResidualBlock(nn.Module):
    """Residual Block.
       Edit architecture in __init__() and forward(). 
       --- channels: 
       --- kernel_size: give a list of kernel sizes along kernels. a list of size 2 specified by size of each kernel.
       --- padding: 
       --- name:
    """
    def __init__(self, in_channels, out_channels, kernel_size = (1, 3), stride = (0, 1), padding = (0, 0), name="1", printForward=False):
        super(ResidualBlock, self).__init__()
        self.printShape = printForward
        self.name = "residual_block_"+str(name)
        self.conv_bn_1 = ConvBn2d(in_channels = in_channels, 
                                  out_channels = out_channels, 
                                  kernel_size = kernel_size[0],
                                  stride = stride[0],
                                  padding = padding[0])
        self.conv_bn_2 = ConvBn2d(in_channels = out_channels, 
                                  out_channels = in_channels, 
                                  kernel_size = kernel_size[1],
                                  stride = stride[1],
                                  padding = padding[1])
        
    def forward(self, x):
        out = self.conv_bn_1(x)
        if self.printShape:
            print(f"{self.name} : {self.conv_bn_1.name} : {out.shape}")
        out = self.conv_bn_2(out)
        if self.printShape:
            print(f"{self.name} : {self.conv_bn_2.name} : {out.shape}")
        out += x
        if self.printShape:
            print(f"{self.name} : {out.shape}")
            self.printShape = False
        return out

class DenoiseNet(nn.Module):
    """Neural Network module.
       Edit architecture in __init__() and forward().
       --- x_channels:
       --- name:
    """
    def __init__(self, x_channels, name = "DenoiseNetwork", printForward=False):
        """
           --- printShape: set boolean value. Print the shape of each layer.
           --- o_channels: set a list of all kernel channels. The size of the list must match the size of the 
                          hidden layers.
        """
        super(DenoiseNet, self).__init__()   
        self.printShape = FLAGS.printShape
        self.name = "DenoiseNetwork"
        out_channels = [int(i) for i in FLAGS.o_channels]
        kernel_sizes = [int(i) for i in FLAGS.kernel_sizes]
        paddings = [int(i) for i in FLAGS.paddings]
        dilations = [int(i) for i in FLAGS.dilations]
        strides = [int(i) for i in FLAGS.strides]

        self.layers_1 = ConvBn2d(in_channels=x_channels, out_channels=out_channels[0], kernel_size=kernel_sizes[0], stride=strides[0], padding=paddings[0], dilation=dilations[0], name="1")
        #self.activation_1 = F.relu
        self.activation_1 = torch.tanh
        self.layers_2 = ConvBn2d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=kernel_sizes[1], stride=strides[1], padding=paddings[1], dilation=dilations[1], name="2")
        #self.activation_2 = F.relu
        self.activation_2 = torch.tanh
        self.layers_3 = ConvBn2d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=kernel_sizes[2], stride=strides[2], padding=paddings[2], dilation=dilations[2], name="3")
        #self.activation_3 = F.relu
        self.activation_3 = torch.tanh
        #self.layers_4 = ConvBn2d(in_channels=out_channels[2], out_channels=x_channels, kernel_size=kernel_sizes[3], stride=strides[3], padding=paddings[3], dilation=dilations[3])
        self.layers_4 = Conv2d(in_channels=out_channels[2], out_channels=x_channels, kernel_size=kernel_sizes[3], stride=strides[3], padding=paddings[3], dilation=dilations[3])
        #self.activation_4 = lambda x: torch.clamp(x, min=0, max=1) #clipped relu
        #self.activation_4 = torch.tanh
        
    def forward(self, x):
        #x = F.normalize(x, p=2)
        if self.printShape:
            print(f"Input : {x.shape}")

        out = self.activation_1(self.layers_1(x))
        if self.printShape:
            print(f"{self.layers_1.name} : {out.shape}")
        
        out = self.activation_2(self.layers_2(out))
        if self.printShape:
            print(f"{self.layers_2.name} : {out.shape}")

        out = self.activation_3(self.layers_3(out))
        if self.printShape:
            print(f"{self.layers_3.name} : {out.shape}")

        out = self.layers_4(out)
        if self.printShape:
            print(f"Output : {out.shape}")
            self.printShape = False

        return out
