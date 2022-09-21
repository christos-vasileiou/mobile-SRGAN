import torch
import torch.nn as nn
from patchgan.Unet import Unet, DepthWiseConvBn2d, Lambda
from absl.flags import FLAGS

##############
# Generators #
##############
class Generator(nn.Module):
    def __init__(self, hps, printForward = False, retain_shape_to_output = True):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            Unet(hps, printForward = printForward, retain_shape_to_output = retain_shape_to_output),
            nn.ReLU()
        )
        initialization(self.main)

    def forward(self, input):
        return self.main(input)

class GeneratorDepthWise(nn.Module):
    def __init__(self, hps, printForward = False, retain_shape_to_output = True):
        super(GeneratorDepthWise, self).__init__()
        self.main = nn.Sequential(
            Unet(hps, printForward = printForward, retain_shape_to_output = retain_shape_to_output, depthwise= True),
            nn.ReLU()
        )
        initialization(self.main)

    def forward(self, input):
        return self.main(input)


class GeneratorPipeline(nn.Module):
    def __init__(self, hps):
        super(GeneratorPipeline, self).__init__()
        ks = hps['ks']
        self.main = nn.Sequential(
            DepthWiseConvBn2d(hps['nc'], hps['nchan'], kernel_size=ks, stride=1, padding=ks//2),
            #nn.Conv2d(hps['nc'], hps['nchan'], kernel_size=ks, stride=1, padding=ks//2),
            nn.ReLU(),
            DepthWiseConvBn2d(hps['nchan'], hps['nchan'] * 2, kernel_size=ks, stride=1, padding=ks//2),
            #nn.Conv2d(hps['nchan'], hps['nchan'] * 2, kernel_size=ks*3, stride=1, padding=(ks*3)//2),
            nn.ReLU(),
            DepthWiseConvBn2d(hps['nchan'] * 2, hps['nchan'] * 4, kernel_size=ks, stride=1, padding=ks//2),
            #nn.Conv2d(hps['nchan'] * 2, hps['nchan'] * 4, kernel_size=ks, stride=1, padding=ks//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hps['nchan'] * 4, out_channels=hps['nc'], kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        initialization(self.main)

    def forward(self, input):
        return self.main(input)




##################
# Discriminators #
##################
class Discriminator(nn.Module):
    def __init__(self, hps):
        super(Discriminator, self).__init__()
        ks4 = 4 if hps['doutput_size'] != 16 else 3
        s4 = 2 if hps['doutput_size'] != 16 else 1
        ks5 = 4 if hps['doutput_size'] != 16 and hps['doutput_size'] != 8 else 3
        s5 = 2 if hps['doutput_size'] != 16 and hps['doutput_size'] != 8 else 1
        self.discriminator_layer_0 = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(hps['nc'], hps['ndf'], 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_1 = nn.Sequential(
            # state size. (ndf) x 128 x 128
            nn.Conv2d(hps['ndf'], hps['ndf'] * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hps['ndf'] * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_2 = nn.Sequential(
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(hps['ndf'] * 2, hps['ndf'] * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hps['ndf'] * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_3 = nn.Sequential(
            # state size. (ndf*4) x 32 x 32
            nn.Conv2d(hps['ndf'] * 4, hps['ndf'] * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hps['ndf'] * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_4 = nn.Sequential(
            # state size. (ndf*8) x 16 x 16
            nn.Conv2d(hps['ndf'] * 8, hps['ndf'] * 16, ks4, stride=s4, padding=1, bias=False),
            nn.BatchNorm2d(hps['ndf'] * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_5 = nn.Sequential(
            # state size. (ndf*16) x 8 x 8
            nn.Conv2d(hps['ndf'] * 16, hps['nc'], ks5, stride=s5, padding=1, bias=False)
        )
        self.discriminator_layer_6 = nn.Sequential(
            nn.Sigmoid()
        )
        initialization(self.discriminator_layer_5)
        initialization(self.discriminator_layer_4)
        initialization(self.discriminator_layer_3)
        initialization(self.discriminator_layer_2)
        initialization(self.discriminator_layer_1)
        initialization(self.discriminator_layer_0)

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        i = 0
        outputs = []
        # print('input: ', x.size())
        layer_name = lambda x: f"discriminator_layer_{x}"
        while hasattr(self, layer_name(i)):
            layer = getattr(self, layer_name(i))
            x = layer(x)
            outputs.append(x)
            # print(f'{x.shape}, {layer}')
            i += 1
        return outputs

class DiscriminatorDepthWise(nn.Module):
    def __init__(self, hps):
        super(DiscriminatorDepthWise, self).__init__()
        ks4 = 4 if hps['doutput_size'] != 16 else 3
        s4 = 2 if hps['doutput_size'] != 16 else 1
        ks5 = 4 if hps['doutput_size'] != 16 and hps['doutput_size'] != 8 else 3
        s5 = 2 if hps['doutput_size'] != 16 and hps['doutput_size'] != 8 else 1
        self.discriminator_layer_0 = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(hps['nc'], hps['ndf'], 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_1 = nn.Sequential(
            # state size. (ndf) x 128 x 128
            DepthWiseConvBn2d(hps['ndf'], hps['ndf'] * 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_2 = nn.Sequential(
            # state size. (ndf*2) x 64 x 64
            DepthWiseConvBn2d(hps['ndf'] * 2, hps['ndf'] * 4, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_3 = nn.Sequential(
            # state size. (ndf*4) x 32 x 32
            DepthWiseConvBn2d(hps['ndf'] * 4, hps['ndf'] * 8, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_4 = nn.Sequential(
            # state size. (ndf*8) x 16 x 16
            DepthWiseConvBn2d(hps['ndf'] * 8, hps['ndf'] * 16, ks4, stride=s4, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator_layer_5 = nn.Sequential(
            # state size. (ndf*16) x 8 x 8
            nn.Conv2d(hps['ndf'] * 16, hps['nc'], ks5, stride=s5, padding=1),
        )
        self.discriminator_layer_6 = nn.Sequential(
            nn.Sigmoid()
        )
        initialization(self.discriminator_layer_5)
        initialization(self.discriminator_layer_4)
        initialization(self.discriminator_layer_3)
        initialization(self.discriminator_layer_2)
        initialization(self.discriminator_layer_1)
        initialization(self.discriminator_layer_0)

    def forward(self, x):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        i=0
        outputs = []
        #print('input: ', x.size())
        layer_name = lambda x: f"discriminator_layer_{x}"
        while hasattr(self, layer_name(i)):
            layer = getattr(self, layer_name(i))
            x = layer(x)
            outputs.append(x)
            #print(f'{i}: {x.shape}')
            i+=1
        return outputs


def initialization(model):
    for i in model.modules():
        if not isinstance(i, nn.modules.container.Sequential):
            classname = i.__class__.__name__
            if hasattr(i, "weight"):
                if classname.find("Conv") != -1:
                    nn.init.normal_(i.weight)
                elif classname.find("BatchNorm2d") != -1:
                    nn.init.normal_(i.weight.data, 1.0, 0.02)
            if hasattr(i, 'bias'):
                #if classname.find("Conv") != -1:
                #    nn.init.zeros_(i.bias)
                if classname.find("BatchNorm2d") != -1:
                    nn.init.constant_(i.bias.data, 0.0)
