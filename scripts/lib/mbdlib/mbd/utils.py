import pymatreader
import torch
from absl.flags import FLAGS
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from os.path import exists
import logging
from mbd.gan_models import *
from torchvision import transforms
import time
import logging

def normalize(x):
    """
    :param x: torch.Tensor or np.Array
    :return: normalized values in the range [0,1]
    """
    return (x-x.min())/(x.max()-x.min())

def PSNR(y_true, y_pred, MAXp=1.):
    """
    Evaluates the PSNR value:
        PSNR = 20 * log10(MAXp) - 10 * log10(MSE).

    :param y_true: ground truth.
    :param y_pred: predicted value.
    :param MAXp: y_true's maximum value of the pixel range (default=1.).
    :return: psnr metric
    """
    if y_true.shape[0] != y_pred.shape[0]:
        raise AssertionError(f"y_true and y_pred must have the same shape")
    if not isinstance(y_true, torch.Tensor) or not isinstance(y_pred, torch.Tensor):
        raise ValueError(f"y_true or y_pred is not a tensor")
    if not isinstance(MAXp, torch.Tensor):
        MAXp = torch.tensor(MAXp)
    z = torch.concat([y_true, y_pred])
    z = normalize(z)
    y_true = z[:z.shape[0]//2]
    y_pred = z[z.shape[0]//2:]
    batch_psnr = 20*torch.log10(MAXp) - 10*torch.log10(torch.mean(torch.square(y_true-y_pred), dim=(-1,-2)))
    return torch.mean(batch_psnr)

def RMSE(y_true, y_pred, eps=1e-6):
    rmse = torch.sqrt(torch.mean(torch.square(y_true-y_pred) + eps, dim=(-1,-2)))
    return torch.mean(rmse)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = RMSE(yhat, y, self.eps)
        return loss

def print_time(ss):
    """
    Print in hh:mm:ss format given duration.
    :param ss: given seconds.
    :return: print in display.
    """
    ss = int(ss)
    hh = 0
    mm = 0
    if ss>(60*60):
        hh = ss//(60*60)
        ss -= (ss//(60*60))*(60*60)
    if ss>60:
        mm = ss//60
        ss -= (ss//60)*60
    hh = str(int(hh)).zfill(2)
    mm = str(int(mm)).zfill(2)
    ss = str(int(ss)).zfill(2)
    logging.info(f"{bcolors.OKGREEN}Training's duration {hh}:{mm}:{ss}{bcolors.ENDC}")

def build_models(hps):
    """ Build the required model

    :param hps: hyperparameters required to build Generator and Discriminator
        - nchan:
            number of channels of the first Conv-Op layer. The number of channels
            for the rest channels depends on that value, Since every next layer
            has twise as much as the previous one.
        - depth:
            Depth of the U-net. This denotes the number of the resolution reduction take place in the U-Net.
        - batch_size:
            batch size during training.
        - ks:
            kernel size of Conv-Op are applied through entire Architecture of the Generator.
        - nc:
            Number of channels in the training images.
        - ndf:
            Size of feature maps in discriminator.
    :return: Generator and Discriminator
    """
    netG=None
    netD=None
    if FLAGS.gentype == 'depthwise':
        netG = GeneratorDepthWise(hps, retain_shape_to_output=FLAGS.retain_shape)
        netD = DiscriminatorDepthWise(hps)
    elif FLAGS.gentype == 'pipeline':
        netG = GeneratorPipeline(hps)
        netD = DiscriminatorDepthWise(hps)
    elif FLAGS.gentype == 'normal':
        netG = Generator(hps, retain_shape_to_output=FLAGS.retain_shape)
        netD = Discriminator(hps)
    return netG, netD

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def plot(data, name='test.pdf', title='', equal=False, aspect3d = (1,1,1)):
    with PdfPages(name) as pdf:
        if len(data.shape) == 3:
            data = data[0]

        parula_map = LinearSegmentedColormap.from_list('parula', _parula_data)

        fig = pyplot.figure(figsize=(15, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        im = ax1.contour((data), cmap=parula_map)
        ax1.set_title('2D ' + title)
        ax1.set_xlabel('x-axis')
        ax1.set_ylabel('y-axis')
        if equal:
            ax1.axis('equal')

        d = np.flip(np.flip(data, axis=0), axis=1)

        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.view_init(0, 0)
        x, y = np.mgrid[:data.shape[0], :data.shape[1]]
        ax2.plot_surface(x, y, d.T, cmap=parula_map, rstride=1, cstride=1, linewidth=0., antialiased=False)
        ax2.set_title('Heatmap 0°-angle View - ' + title)
        ax2.set_ylabel('y-axis')
        ax2.set_zlabel('z-axis')
        if equal:
            ax2.set_box_aspect(aspect = aspect3d)

        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.view_init(30, 45)
        x, y = np.mgrid[:data.shape[0], :data.shape[1]]
        ax3.plot_surface(x, y, d.T, cmap=parula_map, rstride=1, cstride=1, linewidth=0., antialiased=False)
        ax3.set_title('Heatmap 45°-angle View - ' + title)
        ax3.set_xlabel('x-axis')
        ax3.set_ylabel('y-axis')
        ax3.set_zlabel('z-axis')
        if equal:
            ax3.set_box_aspect(aspect = aspect3d)

        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.view_init(90, 90)
        x, y = np.mgrid[:data.shape[0], :data.shape[1]]
        ax4.plot_surface(x, y, d.T, cmap=parula_map, rstride=1, cstride=1, linewidth=0., antialiased=False)
        ax4.set_title('Heatmap 90°-angle View - ' + title)
        ax4.set_ylabel('y-axis')
        ax4.set_xlabel('x-axis')
        if equal:
            ax4.set_box_aspect(aspect = aspect3d)

        pdf.savefig()
        pyplot.close()


def get_name(*argv):
    s = ''
    reduction = ''
    for arg in argv:
        try:
            if arg is not None:
                s += '-'+arg.__class__.__name__
            if arg.reduction:
                reduction = arg.reduction
        except:
            pass
    if 'v2' in FLAGS.dataset:
        s += '-v2'+'-depth'+str(FLAGS.depth)+'-'+reduction
    else:
        s += '-depth'+str(FLAGS.depth)+'-'+reduction

    return s

def plotResults(results):
    fig, [ax1, ax2] = pyplot.subplots(2, 1, sharex = True, figsize = (10, 10))

    ax1.plot(results.index, results["Train Loss"],
             results.index, results["Test Loss"],
             results.index, results["Validation Loss"],
             results.index, results["Min Validation Loss"])
    ax1.legend(["Train", "Test", "Validation", "Min Validation"])
    ax1.set_title("Loss")

    ax2.plot(results.index, results["Validation Loss"]/results["Min Validation Loss"] - 1)
    ax2.set_title("Val Loss/Min Val Loss - 1")

    pyplot.savefig("results.jpg")

def load_model(model, weightspath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if exists(weightspath):
            state_dict = torch.load(
                weightspath,
                map_location=device
            )
            model.load_state_dict(state_dict)
            model = model.to(device)
    except:
        logging.debug(f"{bcolors.FAIL}netG1 couldn't load its trained weights!{bcolors.ENDC}")
    return model

def run_a_sample(hps, training_typo, n_samples=1):
    dataset = pymatreader.read_mat("../../../data/hffh_testing.mat")
    dataset = np.expand_dims(np.stack([dataset["radarImages"], dataset["idealImages"]]).transpose((1, 0, 2, 3)), axis=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG1, _ = build_models(hps)
    netG2, _ = build_models(hps)
    netG1 = load_model(netG1, f'./trained_models/{FLAGS.gentype}/{FLAGS.dataset}/Generator{training_typo}-final.pt')
    netG2 = load_model(netG2, f'./trained_models/{FLAGS.gentype}/{FLAGS.dataset}/Generator{training_typo}.pt')
    logging.info(f'./trained_models/{FLAGS.gentype}/{FLAGS.dataset}/Generator' + training_typo + '.pt')

    preprocess = transforms.Resize([256, 256])
    # dataset[m][n]:
    # --- m -> sample of a dataset,
    # --- n -> 0: SAR image, 1:Ideal image
    logging.info(f"dataset: {dataset.shape}, input: {dataset[0].shape}")
    for i in range(n_samples):
        # detect dataset
        sar_image = np.expand_dims(dataset[i][0], axis=0).astype('float64')
        ideal_image = np.expand_dims(dataset[i][1], axis=0).astype('float64')
        sar_image = preprocess(torch.tensor(sar_image))
        sar_image = sar_image.to(device)
        # feed it to the network
        logging.info(f"{i}")
        try:
            with torch.no_grad():
                netG1 = netG1.to(device)
                output1 = netG1(sar_image).cpu().detach().numpy()
                logging.info(f"output1: min={output1.min()}, max={output1.max()}")
        except:
            logging.debug(f"{bcolors.FAIL}netG1 got an input has shape of {sar_image.shape}{bcolors.ENDC}")
        with torch.no_grad():
            netG2 = netG2.to(device)
            startTime = time.time()
            output2 = netG2(sar_image).cpu().detach().numpy()
            endTime = time.time()
            logging.info(f"output2: min={output2.min()}, max={output2.max()}")
            logging.info(f"Inference in {endTime - startTime} seconds\n\n")

        # logging.info(type(output), output.shape, '\n', output)
        try:
            plot(output1[0][0],
                 name=f"./trained_models/{FLAGS.gentype}/{FLAGS.dataset}/{i}-gan_output1" + training_typo + '-depth' + str(FLAGS.depth) + '_nchan' + str(
                     FLAGS.nchan) + '.pdf', title='Denoised Data')
        except:
            logging.debug(f"{bcolors.FAIL}netG1's plot face an issue{bcolors.ENDC}")

        plot(output2[0][0],
                 name=f"./trained_models/{FLAGS.gentype}/{FLAGS.dataset}/{i}-gan_output2" + training_typo + '-depth' + str(FLAGS.depth) + '_nchan' + str(
                     FLAGS.nchan) + '.pdf', title='Denoised Data')

        # extract contours of inputs and outputs.
        plot(sar_image.cpu().detach().numpy()[0], name=f"./trained_models/{FLAGS.gentype}/{FLAGS.dataset}/{i}-input" + training_typo + '.pdf', title='Radar Data')
        if dataset.shape[1] == 2: # Only Synthetic Dataset contains Ideal
            plot(ideal_image[0], name=f'./trained_models/{FLAGS.gentype}/{FLAGS.dataset}/{i}-ideal' + training_typo + '.pdf', title='Ideal Data')


def convert_size(size):
    if size <1024:
        return size
    elif size < (1024 * 1024):
        return "%.2f KB"%(size/1024)
    elif size < (1024*1024*1024):
        return "%.2f MB"%(size/(1024*1024))
    else:
        return "%.2f GB"%(size/(1024*1024*1024))


def get_trained_type(netG):
    criterion_p2p = None
    criterion_min_max = None
    criterion_perc = None
    criterion_min_max = nn.BCELoss(reduction='none')
    criterion_perc = nn.L1Loss(reduction='sum') if FLAGS.perc else None
    criterion_p2p = nn.L1Loss(reduction='sum') if FLAGS.p2p else None
    training_typo = get_name(netG, criterion_min_max, criterion_perc, criterion_p2p)
    logging.info("training type: ", training_typo)
    return training_typo, criterion_min_max, criterion_perc, criterion_p2p


def run_a_real_sample(netG2, dataset, hps, training_typo, device, n_samples=1):
    if not netG2:
        netG1 = Generator(hps, retain_shape_to_output=FLAGS.retain_shape)
        netG2 = Generator(hps, retain_shape_to_output=FLAGS.retain_shape)
        try:
            if exists(f'./trained_models/{FLAGS.gentype}/{FLAGS.dataset}/GeneratorReal'+str(training_typo)+'-final.pt'):
                state_dict = torch.load(f'./trained_models/{FLAGS.gentype}/{FLAGS.dataset}/GeneratorReal'+str(training_typo)+'-final.pt')
                netG1.load_state_dict(state_dict)
        except:
            pass
        try:
            if exists(f'./trained_models/{FLAGS.gentype}/{FLAGS.dataset}/GeneratorReal'+str(training_typo)+'.pt'):
                state_dict = torch.load(f'./trained_models/{FLAGS.gentype}/{FLAGS.dataset}/GeneratorReal'+str(training_typo)+'.pt')
                netG2.load_state_dict(state_dict)
        except:
            pass
    else:
        netG2.to(torch.device("cpu"))
    # dataset[m]:
    # --- m -> sample of a dataset
    for i in range(n_samples):
        logging.info(f"dataset: {dataset.shape}, input: {dataset[i].shape}")
        sar_image = np.expand_dims(dataset[i], axis=0).astype('float64')
        try:
            output1 = netG1(torch.from_numpy(sar_image))
        except:
            pass
        #try:
        output2 = netG2(torch.from_numpy(sar_image))
        #except:
        #    pass
        # logging.info(f"output: {output.shape}, output: {output.detach().numpy().shape}, {type(output.detach().numpy())}")
        try:
            logging.info(f"\noutput1: {output1.detach().numpy()}")
        except:
            pass
        try:
            logging.info(f"output2: {output2.detach().numpy()}\n")
        except:
            pass
        # extract contours of inputs and outputs.
        plot(dataset[i][0], name=f"./trained_models/{FLAGS.gentype}/{FLAGS.dataset}/{i}-inputReal" + training_typo + '.pdf', title='Input Data')
        plot(dataset[i][1], name=f'./trained_models/{FLAGS.gentype}/{FLAGS.dataset}/{i}-ideal' + training_typo + '.pdf', title='Ideal Data')

        # logging.info(type(output), output.shape, '\n', output)
        try:
            plot(output1.cpu().detach().numpy()[0][0],
                 name=f"./trained_models/{FLAGS.gentype}/{FLAGS.dataset}/{i}-real_output1" + training_typo + '-depth' + str(FLAGS.depth) + '_nchan' + str(
                     FLAGS.nchan) + '.pdf', title='Denoised Data')
        except:
            pass
        #try:
        plot(output2.cpu().detach().numpy()[0][0],
                 name=f"./trained_models/{FLAGS.gentype}/{FLAGS.dataset}/{i}-real_output2" + training_typo + '-depth' + str(FLAGS.depth) + '_nchan' + str(
                     FLAGS.nchan) + '.pdf', title='Denoised Data')

def freeze(model):
    layers = 0
    for p in model.parameters():
        p.requires_grad = False
    return model

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True
    return model

_parula_data = [[0.2081, 0.1663, 0.5292],
                [0.2116238095, 0.1897809524, 0.5776761905],
                [0.212252381, 0.2137714286, 0.6269714286],
                [0.2081, 0.2386, 0.6770857143],
                [0.1959047619, 0.2644571429, 0.7279],
                [0.1707285714, 0.2919380952, 0.779247619],
                [0.1252714286, 0.3242428571, 0.8302714286],
                [0.0591333333, 0.3598333333, 0.8683333333],
                [0.0116952381, 0.3875095238, 0.8819571429],
                [0.0059571429, 0.4086142857, 0.8828428571],
                [0.0165142857, 0.4266, 0.8786333333],
                [0.032852381, 0.4430428571, 0.8719571429],
                [0.0498142857, 0.4585714286, 0.8640571429],
                [0.0629333333, 0.4736904762, 0.8554380952],
                [0.0722666667, 0.4886666667, 0.8467],
                [0.0779428571, 0.5039857143, 0.8383714286],
                [0.079347619, 0.5200238095, 0.8311809524],
                [0.0749428571, 0.5375428571, 0.8262714286],
                [0.0640571429, 0.5569857143, 0.8239571429],
                [0.0487714286, 0.5772238095, 0.8228285714],
                [0.0343428571, 0.5965809524, 0.819852381],
                [0.0265, 0.6137, 0.8135],
                [0.0238904762, 0.6286619048, 0.8037619048],
                [0.0230904762, 0.6417857143, 0.7912666667],
                [0.0227714286, 0.6534857143, 0.7767571429],
                [0.0266619048, 0.6641952381, 0.7607190476],
                [0.0383714286, 0.6742714286, 0.743552381],
                [0.0589714286, 0.6837571429, 0.7253857143],
                [0.0843, 0.6928333333, 0.7061666667],
                [0.1132952381, 0.7015, 0.6858571429],
                [0.1452714286, 0.7097571429, 0.6646285714],
                [0.1801333333, 0.7176571429, 0.6424333333],
                [0.2178285714, 0.7250428571, 0.6192619048],
                [0.2586428571, 0.7317142857, 0.5954285714],
                [0.3021714286, 0.7376047619, 0.5711857143],
                [0.3481666667, 0.7424333333, 0.5472666667],
                [0.3952571429, 0.7459, 0.5244428571],
                [0.4420095238, 0.7480809524, 0.5033142857],
                [0.4871238095, 0.7490619048, 0.4839761905],
                [0.5300285714, 0.7491142857, 0.4661142857],
                [0.5708571429, 0.7485190476, 0.4493904762],
                [0.609852381, 0.7473142857, 0.4336857143],
                [0.6473, 0.7456, 0.4188],
                [0.6834190476, 0.7434761905, 0.4044333333],
                [0.7184095238, 0.7411333333, 0.3904761905],
                [0.7524857143, 0.7384, 0.3768142857],
                [0.7858428571, 0.7355666667, 0.3632714286],
                [0.8185047619, 0.7327333333, 0.3497904762],
                [0.8506571429, 0.7299, 0.3360285714],
                [0.8824333333, 0.7274333333, 0.3217],
                [0.9139333333, 0.7257857143, 0.3062761905],
                [0.9449571429, 0.7261142857, 0.2886428571],
                [0.9738952381, 0.7313952381, 0.266647619],
                [0.9937714286, 0.7454571429, 0.240347619],
                [0.9990428571, 0.7653142857, 0.2164142857],
                [0.9955333333, 0.7860571429, 0.196652381],
                [0.988, 0.8066, 0.1793666667],
                [0.9788571429, 0.8271428571, 0.1633142857],
                [0.9697, 0.8481380952, 0.147452381],
                [0.9625857143, 0.8705142857, 0.1309],
                [0.9588714286, 0.8949, 0.1132428571],
                [0.9598238095, 0.9218333333, 0.0948380952],
                [0.9661, 0.9514428571, 0.0755333333],
                [0.9763, 0.9831, 0.0538]]
