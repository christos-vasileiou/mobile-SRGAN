import torch
import torch.nn as nn
from mobile.Unet import Unet, DepthWiseConvBn2d, Lambda

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

    def forward(self, input):
        return self.main(input)

class GeneratorDepthWise(nn.Module):
    def __init__(self, hps, printForward = False, retain_shape_to_output = True):
        super(GeneratorDepthWise, self).__init__()
        self.main = nn.Sequential(
            Unet(hps, printForward = printForward, retain_shape_to_output = retain_shape_to_output, depthwise= True),
            nn.ReLU()
        )

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

    def forward(self, input):
        return self.main(input)
