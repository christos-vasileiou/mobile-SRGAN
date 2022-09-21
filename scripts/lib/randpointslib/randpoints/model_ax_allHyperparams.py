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
                    'set the learning rate {float}. default: 0.1')

flags.DEFINE_integer('batch_size', 
                    4, 
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

flags.DEFINE_string('load_model',
                    '',
                    'Set a path to a model. default: \'\'')

flags.DEFINE_string('o_name',
                    '',
                    'Set a name for output denoised images. default: \'\'')

class Reshape(nn.Module):
    def __init__(self, name = 1):
        super(Reshape, self).__init__()
        self.name = "Reshape_"+str(name)

    def forward(self, x):
        #print(f"Before reshaping: {x.shape}")
        x = x.view(-1, 1, 256, 256)
        #print(f"After reshaping: {x.shape}")

        return x

class Vectorize(nn.Module):
    def __init__(self, name = 1):
        super(Vectorize, self).__init__()
        self.name = "Vectorize_"+str(name)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        return x

class LinearNormWeight(nn.Module):
    def __init__(self, in_features, out_features, bias = False, name = 1):
        super(LinearNormWeight, self).__init__()
        self.name = "LinearNormWeight_"+str(name)
        self.layer = nn.Linear(in_features, out_features, bias)
        
    def forward(self, x):
        w = self.layer.weight
        w = F.normalize(w, dim=1, p=2)
        x = F.linear(x, w)

        return x

def make_encoder_layers_from_hps(hps, convhps_LUT):
    encoder_layers = []
    i = 0
    while i in range(hps['n_conv_layers']):
        in_channels = 1
        out_channels = 1
        if i >= 1:
            in_channels = hps[f"out_channels_{i-1}"]
        kernel_size = convhps_LUT[hps[f"convhps_{i}"]]['kernel_size']
        padding = convhps_LUT[hps[f"convhps_{i}"]]['padding']
        out_channels = hps[f"out_channels_{i}"]
        if i == hps['n_conv_layers']-1:
            out_channels = 1
        encoder_layers.append(ConvBn2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=padding,
                                        name=str(i)
                                        )
                                )
        if i < hps['n_conv_layers']-1:
            encoder_layers.append(nn.Tanh())
        i = i + 1

    return encoder_layers

class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, name="1", printForward=False):
        super(ConvBn2d, self).__init__()
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

class DenoiseNet_ax(nn.Module):
    """Neural Network module.
        Edit architecture in __init__() and forward().
        --- x_channels:
        --- name:
    """
    def __init__(self, hps, convhps_LUT, encoder_layers = None, printForward = False):
        super(DenoiseNet_ax, self).__init__()
        self.name = "DenoiseNet_ax"
        self.printShape = printForward
        self.hps = hps
        self.convhps_LUT = convhps_LUT
        self.input_shape = (hps['batch_size'], 1, 256, 256)
        
        if encoder_layers == None:
            self.encoder_layers = make_encoder_layers_from_hps(hps, convhps_LUT)
        self.__assign_layers_to_self()
    
    def __create_decoder(self):
        shape = self.input_shape
        x = torch.randn(shape)
        decoder_layers = []
        with torch.no_grad():
            for layer in self.encoder_layers:
                x = layer(x)
        
        print(x.shape[1])
        print(self.input_shape[-1]*self.input_shape[-2])
        print()

        decoder_layers.append(LinearNormWeight(x.shape[1], self.input_shape[-1]*self.input_shape[-2], bias=False))
        decoder_layers.append(Reshape())

        return decoder_layers

    def __assign_layers_to_self(self):
        for i, layer in enumerate(self.encoder_layers):
            layer_name = f"encoder_layer_{i}"
            setattr(self, layer_name, layer)

        #for i, layer in enumerate(self.decoder_layers):
        #    layer_name = f"decoder_layer_{i}"
        #    setattr(self, layer_name, layer)
        #    
        #    if isinstance(layer, nn.Linear) or isinstance(layer, LinearNormWeight):
        #        self.fc_layer_name = layer_name

    def get_embedding(self, x):
        i = 0
        layer_name = lambda x: f"encoder_layer_{x}"
        while hasattr(self, layer_name(i)):
            layer = getattr(self, layer_name(i))
            x = layer(x)
            #if i < self.hps['n_conv_layers'] and self.printShape and i%2==0:
            #    print(f"{layer.name}, {x.shape}")
            i += 1

        #i = 0
        #layer_name = lambda x: f"decoder_layer_{x}"
        #while hasattr(self, layer_name(i)):
        #    layer = getattr(self, layer_name(i))
        #    if self.printShape:
        #        #print(f"{layer.name}, {x.shape}")
        #        self.printShape = False
        #    x = layer(x)
        #    i += 1

        return x

    def forward(self, x):
        out = self.get_embedding(x)
        return out


def get_criterion_from_hps(hps):
    if hps['criterion'] == 'GaussianNLLLoss':
        return nn.GaussianNLLLoss()
    elif hps['criterion'] == 'MSELoss':
        return nn.MSELoss()
    else:
        raise Exception("NYI")

def get_optimizer_from_hps(hps, net):
    if hps['optimizer'] == 'Adam': 
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=hps['lr'],
                                     weight_decay=hps['weight_decay'])
    elif hps['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=hps['lr'],
                                    weight_decay=hps['weight_decay'])
    else:
        raise Exception("NYI")
    return optimizer


def run_a_sample_ax(model, validset, testset, device, ax_trial):
    # dataset[m][n]: 
    # --- m -> sample of a dataset, 
    # --- n -> 0: SAR image, 1:Ideal image
    print(f"input: {validset[0][0].shape}")
    sar_image = np.expand_dims(validset[0][0], axis=0)
    output = model(torch.from_numpy(sar_image).to(device)).to(device)
    #print(f"output: {output.shape}, output: {output.detach().numpy().shape}, {type(output.detach().numpy())}")
    #print(f"output: {output.detach().numpy()}\n")
    
    # extract contours of inputs and outputs.
    pyplot.figure()
    pyplot.contour((validset[0][0][0]))
    pyplot.title (f"SAR Data")
    pyplot.savefig(f"input-valid.png")

    pyplot.figure()
    pyplot.contour((output.cpu().detach().numpy()[0][0]))
    pyplot.title (f"Denoised Data")
    pyplot.savefig(ax_trial+"-valid.png")


    sar_image = np.expand_dims(testset[0][0], axis=0)
    output = model(torch.from_numpy(sar_image).to(device)).to(device)
    #print(f"output: {output.shape}, output: {output.detach().numpy().shape}, {type(output.detach().numpy())}")
    #print(f"output: {output.detach().numpy()}\n")
    
    # extract contours of inputs and outputs.
    pyplot.figure()
    pyplot.contour((testset[0][0][0]))
    pyplot.title (f"SAR Data")
    pyplot.savefig(f"input-test.png")

    pyplot.figure()
    pyplot.contour((output.cpu().detach().numpy()[0][0]))
    pyplot.title (f"Denoised Data")
    pyplot.savefig(ax_trial+"-"+FLAGS.o_name+"-test.png")
