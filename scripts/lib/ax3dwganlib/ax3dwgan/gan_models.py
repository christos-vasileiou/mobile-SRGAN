from absl import flags, app
from absl.flags import FLAGS
import torch
import torch.nn as nn
from ax3dwgan.Unet import Unet3d, DepthWiseConvBn3d, Lambda
from ax3dwgan.minibatch_discrimination import MiniBatchDiscrimination

#############
# Generators #
##############
class Generator3d(nn.Module):
    def __init__(self, hps, printForward = False, retain_shape_to_output = True):
        super(Generator3d, self).__init__()
        self.main = nn.Sequential(
            Unet3d(hps, printForward = printForward, retain_shape_to_output = retain_shape_to_output),
            nn.ReLU()
        )
        initialization(self.main)

    def forward(self, input):
        return self.main(input)

class GeneratorDepthWise3d(nn.Module):
    def __init__(self, hps, printForward = False, retain_shape_to_output = True):
        super(GeneratorDepthWise3d, self).__init__()
        self.main = nn.Sequential(
            Unet3d(hps, printForward = printForward, retain_shape_to_output = retain_shape_to_output, depthwise = True),
            nn.ReLU()
        )
        initialization(self.main)

    def forward(self, input):
        return self.main(input)

class GeneratorPipeline3d(nn.Module):
    def __init__(self, hps):
        super(GeneratorPipeline3d, self).__init__()
        ks = hps['ks']
        self.main = nn.Sequential(
            DepthWiseConvBn3d(hps['nc'], hps['nchan'], kernel_size=ks, stride=1, padding=ks//2),
            nn.LeakyReLU(0.2),
            DepthWiseConvBn3d(hps['nchan'], hps['nchan'] * 2, kernel_size=ks*3, stride=1, padding=ks*3//2),
            nn.LeakyReLU(0.2),
            DepthWiseConvBn3d(hps['nchan'] * 2, hps['nchan'] * 4, kernel_size=ks, stride=1, padding=ks//2),
            nn.LeakyReLU(0.2),
            DepthWiseConvBn3d(hps['nchan'] * 4, hps['nc'], kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        initialization(self.main)

    def forward(self, input):
        return self.main(input)




##################
# Discriminators #
##################
class Discriminator3d(nn.Module):
    def __init__(self, hps):
        super(Discriminator3d, self).__init__()
        ks3 = 4 if hps['doutput_size'] != 8 else 3
        s3 = 2 if hps['doutput_size'] != 8 else 1
        ks4 = 4 if hps['doutput_size'] != 8 and hps['doutput_size'] != 4 else 3
        s4 = 2 if hps['doutput_size'] != 8 and hps['doutput_size'] != 4 else 1
        ks5 = 4 if hps['doutput_size'] != 8 and hps['doutput_size'] != 4 and hps['doutput_size'] != 2 else 3
        s5 = 2 if hps['doutput_size'] != 8 and hps['doutput_size'] != 4 and hps['doutput_size'] != 2 else 1
        self.discriminator_layer_0 = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv3d(hps['nc'], hps['ndf'], 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )
        self.discriminator_layer_1 = nn.Sequential(
            # state size. (ndf) x 128 x 128
            nn.Conv3d(hps['ndf'], hps['ndf'] * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(hps['ndf'] * 2),
            nn.LeakyReLU(0.2)
        )
        self.discriminator_layer_2 = nn.Sequential(
            # state size. (ndf*2) x 64 x 64
            nn.Conv3d(hps['ndf'] * 2, hps['ndf'] * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(hps['ndf'] * 4),
            nn.LeakyReLU(0.2)
        )
        self.discriminator_layer_3 = nn.Sequential(
            # state size. (ndf*4) x 32 x 32
            nn.Conv3d(hps['ndf'] * 4, hps['ndf'] * 8, ks3, stride=s3, padding=1, bias=False),
            nn.BatchNorm3d(hps['ndf'] * 8),
            nn.LeakyReLU(0.2)
        )
        self.discriminator_layer_4 = nn.Sequential(
            # state size. (ndf*8) x 16 x 16
            nn.Conv3d(hps['ndf'] * 8, hps['ndf'] * 16, ks4, stride=s4, padding=1, bias=False),
            nn.BatchNorm3d(hps['ndf'] * 16),
            nn.LeakyReLU(0.2)
        )
        self.discriminator_layer_5 = nn.Sequential(
            # state size. (ndf*16) x 8 x 8
            nn.Conv3d(hps['ndf'] * 16, hps['nc'], ks5, stride=s5, padding=1, bias=False),
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
        i=0
        outputs = []
        #print('input: ', x.size())
        layer_name = lambda x: f"discriminator_layer_{x}"
        while hasattr(self, layer_name(i)):
            layer = getattr(self, layer_name(i))
            x = layer(x.to(torch.device("cuda")))
            #print(f"{i} out: {x.size()}: {layer}")
            outputs.append(x)
            i+=1
        return outputs

class DiscriminatorDepthWise3d(nn.Module):
    def __init__(self, hps):
        super(DiscriminatorDepthWise3d, self).__init__()
        ks3 = 4 if hps['doutput_size'] != 8 else 3
        s3 = 2 if hps['doutput_size'] != 8 else 1
        ks4 = 4 if hps['doutput_size'] != 8 and hps['doutput_size'] != 4 else 3
        s4 = 2 if hps['doutput_size'] != 8 and hps['doutput_size'] != 4 else 1
        ks5 = 4 if hps['doutput_size'] != 8 and hps['doutput_size'] != 4 and hps['doutput_size'] != 2 else 3
        s5 = 2 if hps['doutput_size'] != 8 and hps['doutput_size'] != 4 and hps['doutput_size'] != 2 else 1
        self.discriminator_layer_0 = nn.Sequential(
            # input is (nc) x 64 x 64 x 64
            nn.Conv3d(hps['nc'], hps['ndf'], 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )
        self.discriminator_layer_1 = nn.Sequential(
            # state size. (ndf) x 32 x 32 x 32
            DepthWiseConvBn3d(hps['ndf'], hps['ndf'] * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.discriminator_layer_2 = nn.Sequential(
            # state size. (ndf*2) x 16 x 16 x 16
            DepthWiseConvBn3d(hps['ndf'] * 2, hps['ndf'] * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.discriminator_layer_3 = nn.Sequential(
            # state size. (ndf*4) x 8 x 8 x 8
            DepthWiseConvBn3d(hps['ndf'] * 4, hps['ndf'] * 8, kernel_size=ks3, stride=s3, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.discriminator_layer_4 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4 x 4
            DepthWiseConvBn3d(hps['ndf'] * 8, hps['ndf'] * 16, kernel_size=ks4, stride=s4, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.discriminator_layer_5 = nn.Sequential(
            # state size. (ndf*16) x 8 x 8
            nn.Conv3d(hps['ndf'] * 16, hps['nc'], kernel_size=ks5, stride=s5, padding=1),
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
        i=0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #outputs = torch.tensor([]).to(device)
        outputs = []
        #print('input: ', x.size())
        layer_name = lambda x: f"discriminator_layer_{x}"
        while hasattr(self, layer_name(i)):
            layer = getattr(self, layer_name(i))
            layer = layer.to(device)
            x = layer(x.to(device)).to(device)
            #print(f"{i} out: {x.size()}, {layer_name(i)}")
            outputs.append(x)
            #outputs = torch.concat([outputs, torch.unsqueeze(x,0)], 0)
            i+=1

        #assert (outputs[-1]==x).all() # confirm that the concatenation is in Round-Robin fashion. Last concatenation is the same with the last element
        return outputs


class DiscriminatorDepthWise3dMBD(nn.Module):
    def __init__(self, hps):
        super(DiscriminatorDepthWise3dMBD, self).__init__()
        self.discriminator_layer_0 = nn.Sequential(
            # input is (nc) x 64 x 64 x 64
            nn.Conv3d(hps['nc'], hps['ndf'], 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )
        self.discriminator_layer_1 = nn.Sequential(
            # state size. (ndf) x 32 x 32 x 32
            DepthWiseConvBn3d(hps['ndf'], hps['ndf'] * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.discriminator_layer_2 = nn.Sequential(
            # state size. (ndf*2) x 16 x 16 x 16
            DepthWiseConvBn3d(hps['ndf'] * 2, hps['ndf'] * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.discriminator_layer_3 = nn.Sequential(
            # state size. (ndf*4) x 8 x 8 x 8
            DepthWiseConvBn3d(hps['ndf'] * 4, hps['ndf'] * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.discriminator_layer_4 = nn.Sequential(
            # state size. (ndf*16) x 4 x 4 x 4
            Lambda(lambda x: x.reshape((x.shape[0], -1))),
            nn.Linear(hps['ndf']*8 * (4*4*4), (hps['doutput_size']**3))
        )
        self.mbd = MiniBatchDiscrimination(hps['doutput_size']**3, hps['doutput_size']**3, 100)
        self.discriminator_layer_5 = nn.Sequential(
            # state size. batch_size x (dos*dos*dos + dos*dos*dos)
            Lambda(lambda x: x.reshape((x.shape[0], 2, hps['doutput_size'], hps['doutput_size'], hps['doutput_size']))),
            nn.Conv3d(2, hps['nc'], kernel_size=4, stride=2, padding=1)
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
        i=0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        outputs = []
        #print('input: ', x.size())
        layer_name = lambda x: f"discriminator_layer_{x}"
        while hasattr(self, layer_name(i)):
            if i == 5:
                xx = self.mbd(x, x.size(0))
                x = torch.cat((x.to(device), xx.to(device)), dim=1).float()
                #print(f'{x.shape}, {self.mbd.__class__.__name__}')
                layer = getattr(self, layer_name(i))
                x = layer(x)
                #print(f'{x.shape}, {layer_name(i)}')
                outputs.append(x)
            else:
                layer = getattr(self, layer_name(i))
                x = layer(x)
                outputs.append(x)
                #print(f'{x.shape}, {layer_name(i)}')
            i += 1
        #assert (outputs[-1]==x).all() # confirm that the concatenation is in Round-Robin fashion. Last concatenation is the same with the last element
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
