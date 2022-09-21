from absl import flags, app
from absl.flags import FLAGS
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler
import random
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot

flags.DEFINE_float('learning_rate',
                   0.1,
                   'set the learning rate {float}.')

flags.DEFINE_integer('batch_size',
                     16,
                     'set the batch size {integer}.')

flags.DEFINE_integer('n_epochs',
                     5,
                     'set the number of epochs {integer}.')

flags.DEFINE_float('schedFactor',
                   0.1,
                   'set the factor of the scheduler {float}.')

flags.DEFINE_integer('schedPatience',
                     3,
                     'set the patience of the scheduler {integer}.')

flags.DEFINE_float('weight_decay',
                   0.14,
                   'set the weight decay {float}.')

flags.DEFINE_boolean('printShape',
                     False,
                     'set if the shape of the layers are being printed {boolean}.')

flags.DEFINE_integer('kernel_size',
                     3,
                     'set the size of kernel for the Generator. .')

flags.DEFINE_integer('nchan',
                     64,
                     'set a number of channels (feature maps) for the Generator. The number of channels are doubled as we go deeper to the Encoder of the Unet. On the other hand the number of channels are halved as we go up to the Decoder of the Unet. The last layer is hardcoded to 1.')

flags.DEFINE_integer('ndf',
                     64,
                     'set a number of channels (feature maps) for the Discriminator.')

flags.DEFINE_integer('depth',
                     2,
                     'set a number of the Unet\'s depth.')

flags.DEFINE_string('o_name',
                    '',
                    'Set a name for output denoised images. default: \'\'')

flags.DEFINE_boolean('retain_shape',
                     True,
                     'set if the shape of the output resolution of the Unet will be the same as input.')


class DepthWiseConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthWiseConvBn2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        return self.bn(self.conv2(self.conv1(x)))

class UnetBottleNeck(nn.Module):
    def __init__(self, in_channels, inner_nch, outer_nch, kernel_size=3, retain_shape_to_output=True, depthwise=False, name="1", printForward=True,
                 bn=True):
        super(UnetBottleNeck, self).__init__()
        self.name = "unet_bottleneck_" + str(name)
        self.printShape = printForward
        if bn:
            # self.conv1 = ConvInstNorm2d(in_channels = in_channels,
            #                      out_channels = inner_nch,
            #                      kernel_size = 3,
            #                      stride = 1,
            #                      padding = 1 if retain_shape_to_output else 0)
            if depthwise:
                self.conv1 = DepthWiseConvBn2d(in_channels=in_channels,
                                      out_channels=inner_nch,
                                      kernel_size=kernel_size,
                                      stride=1,
                                      padding=kernel_size//2 if retain_shape_to_output else 0)
            else:
                self.conv1 = ConvBn2d(in_channels=in_channels,
                                      out_channels=inner_nch,
                                      kernel_size=kernel_size,
                                      stride=1,
                                      padding=kernel_size//2 if retain_shape_to_output else 0)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels,
                                   out_channels=inner_nch,
                                   kernel_size=kernel_size,
                                   stride=1,
                                   padding=kernel_size//2 if retain_shape_to_output else 0)
        self.func1 = nn.ReLU()
        if bn:
            # self.conv2 = ConvInstNorm2d(in_channels = inner_nch,
            #                      out_channels = outer_nch,
            #                      kernel_size = 3,
            #                      stride = 1,
            #                      padding = 1 if retain_shape_to_output else 0)
            if depthwise:
                self.conv2 = DepthWiseConvBn2d(in_channels=inner_nch,
                                      out_channels=outer_nch,
                                      kernel_size=kernel_size,
                                      stride=1,
                                      padding=kernel_size//2 if retain_shape_to_output else 0)
            else:
                self.conv2 = ConvBn2d(in_channels=inner_nch,
                                      out_channels=outer_nch,
                                      kernel_size=kernel_size,
                                      stride=1,
                                      padding=kernel_size//2 if retain_shape_to_output else 0)
        else:
            self.conv2 = nn.Conv2d(in_channels=inner_nch,
                                   out_channels=outer_nch,
                                   kernel_size=kernel_size,
                                   stride=1,
                                   padding=kernel_size//2 if retain_shape_to_output else 0)
        self.func2 = nn.ReLU()
        # self.model = nn.Sequential([conv1, func1, conv2, func2])

    def forward(self, x):
        x = self.func1(self.conv1(x))
        x = self.func2(self.conv2(x))
        return x


class ConvInstNorm2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, name="1",
                 printForward=False):
        super(ConvInstNorm2d, self).__init__()
        self.name = "conv_instnorm_2d_" + str(name)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation)
        self.instnorm = nn.InstanceNorm2d(num_features=out_channels, track_running_stats=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.instnorm(out)
        return out


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, name="1",
                 printForward=False):
        super(ConvBn2d, self).__init__()
        self.name = "conv_bn_2d_" + str(name)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
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

    def __init__(self, in_channels, out_channels, kernel_size=(1, 3), stride=(0, 1), padding=(0, 0), name="1",
                 printForward=False):
        super(ResidualBlock, self).__init__()
        self.printShape = printForward
        self.name = "residual_block_" + str(name)
        self.conv_bn_1 = ConvBn2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size[0],
                                  stride=stride[0],
                                  padding=padding[0])
        self.conv_bn_2 = ConvBn2d(in_channels=out_channels,
                                  out_channels=in_channels,
                                  kernel_size=kernel_size[1],
                                  stride=stride[1],
                                  padding=padding[1])

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


def make_encoder_layers_from_hps(hps, retain_shape_to_output=True, depthwise=False):
    """ Create the layers of the Unet.
        - hps:
            -- depth: deepest Unet bottleneck. It is placed after the encoder and before the decoder.
            -- nchan: number of channels of the "first bottleneck". "First Bottleneck" is considered the first bottleneck of the Unet. 
                         As we go deeper to the encoder the number of channels doubles.
            -- 
    """

    # the deepest bottleneck consists of: initial channels * 2 ^ deepest BottleNeck.
    model = UnetBottleNeck(in_channels=hps['nchan'] * 2 ** (hps['depth'] - 1),
                           inner_nch=hps['nchan'] * 2 ** hps['depth'],
                           outer_nch=hps['nchan'] * 2 ** (hps['depth'] - 1),
                           kernel_size=hps['ks'],
                           retain_shape_to_output=retain_shape_to_output,
                           depthwise=depthwise,
                           name=str(hps['depth']))

    for i in reversed(range(hps['depth'])):
        encoder = UnetBottleNeck(in_channels=int(hps['nchan'] * 2 ** (i - 1)),
                                 inner_nch=int(hps['nchan'] * 2 ** i),
                                 outer_nch=int(hps['nchan'] * 2 ** i),
                                 kernel_size=hps['ks'],
                                 retain_shape_to_output=retain_shape_to_output,
                                 depthwise=depthwise,
                                 name="encoder_" + str(i))
        down = DepthWiseConvBn2d(int(hps['nchan'] * 2 ** i), int(hps['nchan'] * 2 ** i), kernel_size=2, stride=2, padding=0) if depthwise else ConvBn2d(int(hps['nchan'] * 2 ** i), int(hps['nchan'] * 2 ** i), kernel_size=2, stride=2, padding=0)
        #down = nn.MaxPool2d(kernel_size=2,
        #                    stride=2
        #                    )
        up = nn.ConvTranspose2d(in_channels=int(hps['nchan'] * 2 ** i),
                                out_channels=int(hps['nchan'] * 2 ** i),
                                kernel_size=2,
                                stride=2,
                                padding=0
                                )
        decoder = UnetBottleNeck(in_channels=int(hps['nchan'] * 2 ** (i + 1)),
                                 inner_nch=int(hps['nchan'] * 2 ** i),
                                 outer_nch=int(hps['nchan'] * 2 ** (i - 1)),
                                 kernel_size=hps['ks'],
                                 retain_shape_to_output=retain_shape_to_output,
                                 depthwise=depthwise,
                                 name="decoder_" + str(i))
        if i == 0:
            encoder = UnetBottleNeck(in_channels=1,
                                     inner_nch=int(hps['nchan']),
                                     outer_nch=int(hps['nchan']),
                                     kernel_size=hps['ks'],
                                     retain_shape_to_output=retain_shape_to_output,
                                     depthwise=False,
                                     name="encoder_" + str(i)
                                     )
            decoder = UnetBottleNeck(in_channels=int(hps['nchan'] * 2 ** (i + 1)),
                                     inner_nch=int(hps['nchan'] * 2 ** i),
                                     outer_nch=int(hps['nchan'] * 2 ** i),
                                     kernel_size=hps['ks'],
                                     retain_shape_to_output=retain_shape_to_output,
                                     depthwise=depthwise,
                                     name="decoder_" + str(i)
                                     )
            outermost = nn.Conv2d(in_channels=int(hps['nchan'] * 2 ** i),
                                  out_channels=1,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
            model = [encoder] + [down] + [model] + [up] + [decoder] + [outermost]
        else:
            model = [encoder] + [down] + [model] + [up] + [decoder]

    return model


class Unet(nn.Module):
    """Neural Network module.
        Edit architecture in __init__() and forward().
        --- x_channels:
        --- name:
    """

    def __init__(self, hps, printForward=False, retain_shape_to_output=True, depthwise=False):
        super(Unet, self).__init__()
        self.name = "Unet"
        self.printShape = printForward
        self.hps = hps

        self.layers = make_encoder_layers_from_hps(hps, retain_shape_to_output, depthwise=depthwise)
        # print(self.layers, '\n')
        self.__assign_layers_to_self()

    # I need to fix that!
    def __assign_layers_to_self(self):

        def assign_layers(self, layers, i):
            if isinstance(layers, list):
                for layer in layers:
                    i = assign_layers(self, layer, i)
            else:
                layer_name = f"unet_layer_{i}"
                setattr(self, layer_name, layers)
                # print(f"{i} - {layers}")
                i += 1
            return i

        i = 0
        assign_layers(self, self.layers, i)

    def get_embedding(self, x):
        i = 0
        cropped = []
        layer_name = lambda x: f"unet_layer_{x}"
        while hasattr(self, layer_name(i)):
            layer = getattr(self, layer_name(i))
            # 2*depth: number of layers of the encoder. Bottlenecks and MaxPools.
            if i < 2 * self.hps['depth'] and i % 2 == 0:
                x = layer(x)
                cropped.append(x)
            elif i > 2 * self.hps['depth'] and i % 2 == 0:
                paste = torch.cat([cropped.pop(-1)[:x.shape[0], :x.shape[1], :x.shape[2], :x.shape[3]], x], axis=1)
                x = layer(paste)
            else:
                x = layer(x)
            i += 1
        return x

    def forward(self, x):
        # print('#################### Forward Passing ####################')
        out = self.get_embedding(x)
        return out
