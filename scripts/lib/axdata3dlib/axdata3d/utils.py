from absl.flags import FLAGS
import numpy as np
import plotly.graph_objs as go
import torch
import torch.nn as nn
import logging
import os
from os.path import exists
from axdata3d.gan_models import Generator3d, GeneratorDepthWise3d, GeneratorPipeline3d, DiscriminatorDepthWise3d, Discriminator3d, DiscriminatorDepthWise3dMBD
import torchio as tio
import time
import pymatreader
from enum import Enum

def get_time(ss):
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
    return f"{bcolors.OKGREEN}Training's duration {hh}:{mm}:{ss}{bcolors.ENDC}"


def count_samples():
    i = 0
    while exists(f"../../../data/data3d/data3d-{i}.mat"):
            i+=1
    return i

class Mode(Enum):
    training = 1
    testing = 2

def hps_initialization():
    depth = FLAGS.depth
    kernel_size = FLAGS.kernel_size
    nchan = FLAGS.nchan
    batch_size = FLAGS.batch_size
    ndf = FLAGS.ndf
    n_epochs = FLAGS.n_epochs
    doutput_size = FLAGS.doutput_size
    learning_rateG = 0.007105769
    learning_rateD = 0.006586546
    weight_decayG = 0.045307002
    weight_decayD = 0.093181123
    perc_lambda_1 = 0.001368283
    perc_lambda_2 = 0.148762791
    perc_lambda_3 = 0.185623507
    perc_lambda_4 = 0.001406883
    perc_lambda_5 = 0.078764838
    pixel_lambda = 1562.366512

    hps = {
        'depth': depth,
        'nchan': nchan,
        # batch size during training
        'batch_size': batch_size,
        # kernel size of Conv-Op. For Pipeline is applied for the first layer.
        'ks': kernel_size,
        # Number of workers for dataloader
        'workers': 2,
        # Number of channels in the training images.
        'nc': 1,
        # Size of feature maps in discriminator
        'ndf': ndf,
        # Patch size. Discriminator's output size.
        'doutput_size': doutput_size,
        # Beta1 hyperparam for Adam optimizers
        'beta1': 0.5,
        # Number of GPUs available. Use 0 for CPU mode.
        'ngpu': 1,  # torch.cuda.device_count(),
        'n_epochs': n_epochs,
        'learning_rateG': learning_rateG,
        'learning_rateD': learning_rateD,
        'weight_decayG': weight_decayG,
        'weight_decayD': weight_decayD,
        'perc_lambda_1': perc_lambda_1,
        'perc_lambda_2': perc_lambda_2,
        'perc_lambda_3': perc_lambda_3,
        'perc_lambda_4': perc_lambda_4,
        'perc_lambda_5': perc_lambda_5,
        'pixel_lambda': pixel_lambda,
        'trial_time': 0
    }
    return hps

def normalize(x):
    """
    :param x: torch.Tensor or np.Array
    :return: normalized values in the range [0,1]
    """
    preprocess = tio.Resize([64, 64, 64])
    processed = torch.tensor([]).cpu()
    if len(x.shape) > 4:
        x = x.detach().cpu()
        for i in x:
            processed = torch.concat([processed, preprocess(i)])
    x = processed
    #min = torch.unsqueeze(torch.unsqueeze(x.min(dim=-1)[0].min(dim=-1)[0], dim=-1), dim=-1)
    #max = torch.unsqueeze(torch.unsqueeze(x.max(dim=-1)[0].max(dim=-1)[0], dim=-1), dim=-1)
    #min = torch.unsqueeze(x.min(dim=-1)[0].min(dim=-1)[0], dim=-1)
    #max = torch.unsqueeze(x.max(dim=-1)[0].max(dim=-1)[0], dim=-1)
    return (x-x.min())/(x.max()-x.min())

def PSNR(y_true, y_pred):
    """
    Evaluates the PSNR value:
        PSNR = 20 * log10(MAXp) - 10 * log10(MSE).

    :param y_true: ground truth.
    :param y_pred: predicted value.
    :param MAXp: y_true's maximum value of the pixel range (default=1.).
    :return: psnr metric
    """
    """
    if y_true.shape[0] != y_pred.shape[0]:
        raise AssertionError(f"y_true and y_pred must have the same shape")
    if not isinstance(y_true, torch.Tensor) or not isinstance(y_pred, torch.Tensor):
        raise ValueError(f"y_true or y_pred is not a tensor")
    """
    #z = torch.concat([y_true, y_pred])
    #z = normalize(z)
    #y_true = z[:z.shape[0]//2]
    #y_pred = z[z.shape[0]//2:]

    #y_true = normalize(y_true)
    #y_pred = normalize(y_pred)
    MAXp = torch.tensor(1.)
    if y_true.max() != 1.:
        MAXp = torch.tensor(y_true.max())

    batch_psnr = 20*torch.log10(MAXp)-10*torch.log10(torch.mean(torch.square(y_true-y_pred), dim=(-1,-2)))
    return batch_psnr

def RMSE(y_true, y_pred, eps=1e-6):
    #z = torch.concat([y_true, y_pred])
    #z = normalize(z)
    #y_true = z[:z.shape[0]//2]
    #y_pred = z[z.shape[0]//2:]
    #y_true = normalize(y_true)
    #y_pred = normalize(y_pred)
    rmse = torch.sqrt(torch.mean(torch.square(y_true-y_pred) + eps, dim=(-1,-2)))
    return rmse

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = RMSE(yhat, y, self.eps)
        return loss

def build_models(hps):
    netG=None
    netD=None
    if FLAGS.gentype == 'depthwise':
        netG = GeneratorDepthWise3d(hps, retain_shape_to_output=FLAGS.retain_shape)
        netD = DiscriminatorDepthWise3d(hps)
    elif FLAGS.gentype == 'pipeline':
        netG = GeneratorPipeline3d(hps)
        netD = DiscriminatorDepthWise3d(hps)
    elif FLAGS.gentype == 'mbd':
        netG = GeneratorDepthWise3d(hps)
        netD = DiscriminatorDepthWise3dMBD(hps)
    elif FLAGS.gentype == 'normal':
        netG = Generator3d(hps, retain_shape_to_output=FLAGS.retain_shape)
        netD = Discriminator3d(hps)
    return netG, netD

def run_handler(net, inputs):
    out = torch.tensor([])
    oom = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        out = net(inputs)
    except RuntimeError:  # Out of memory
        oom = True
    if oom:
        #print(inputs.shape, net.__class__.__name__)
        out = out.to(device)
        for i in range(inputs.shape[0]):
            d = torch.unsqueeze(inputs[i], 0).to(device)
            d = net(d)
            if "Generator" in net.__class__.__name__:
                out = torch.concat([out, d])
            elif "Discriminator" in net.__class__.__name__:
                out = torch.concat([out, d])

    return out

def plot_3d(data, filename):
    """
    :param data: 3D data to plot.
    :param filename: define the name of the html filename.
    :return
        Plot and export the html file
    """
    logging.info(f"{filename} -> Shape: {data.shape} # MIN: {data.min()} and MAX: {data.max()}")
    isorange = (data.max()-data.min())
    isomin = data.min() + isorange * 0.05
    isomax = data.min() + isorange
    X,Y,Z = np.mgrid[:data.shape[0], :data.shape[1], :data.shape[2]]
    volume = go.Volume(x=X.flatten(),
             y=Y.flatten(),
             z=Z.flatten(),
             value=data.flatten(),
             isomin=isomin,
             isomax=isomax,
             surface_count=13,
             opacity=0.5
    )
    fig = go.Figure(data=volume)
    # script mode
    fig.write_html(filename)
    fig.show()

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
            model.float()
    except:
        logging.debug(f"{bcolors.FAIL}netG1 couldn't load its trained weights!{bcolors.ENDC}")
    return model

def run_a_3d_sample(netG2, features, labels, hps, training_typo, device, n_samples=1, ax_trial='Train-3d'):
    if features is None:
        features = []
        labels = []
        for i in range(n_samples):
            features.append(pymatreader.read_mat(f'../../../data/data3d/data3d-{i}.mat')['features'])
            labels.append(pymatreader.read_mat(f'../../../data/data3d/data3d-{i}.mat')['labels'])
    features = np.array(features)
    labels = np.array(labels)
    preprocess = tio.Resize([FLAGS.input_size for _ in range(3)])
    postprocess = tio.Resize([128 for _ in range(3)])
    dataset = preprocess(torch.tensor(features)).numpy()
    labels = preprocess(torch.tensor(labels)).numpy()
    if not netG2:
        netG1, _ = build_models(hps)
        netG2, _ = build_models(hps)
        netG1 = load_model(netG1,f'./trained_models/AxTrial-{ax_trial}/Generator3d{training_typo}-final.pt')
        netG2 = load_model(netG2,f'./trained_models/AxTrial-{ax_trial}/Generator3d{training_typo}.pt')
        logging.info(f'./trained_models/AxTrial-{ax_trial}/Generator3d' + training_typo + '.pt')

    # dataset[m][n]:
    # --- m -> sample of a dataset,
    # --- n -> 0: SAR image, 1:Ideal image
    for i in range(n_samples):
        # detect dataset
        logging.info(f"dataset: {dataset.shape}, labels: {labels.shape}")
        sar_image = np.expand_dims(np.expand_dims(dataset[i], axis=0), axis=0).astype('float32')
        label = np.expand_dims(np.expand_dims(labels[i], axis=0), axis=0).astype('float32')
        logging.info(f"Sar Image: {sar_image.shape} {sar_image.dtype}, Label: {label.shape} {label.dtype}")
        # feed it to the network
        try:
            sar_image = torch.from_numpy(sar_image)
            label = torch.from_numpy(label)
            with torch.no_grad():
                netG1 = netG1.to(device)
                output1 = netG1(sar_image.to(device))
                logging.info(f"output1: min={output1.min()}, max={output1.max()}")
        except:
            logging.debug(f"Error: netG1 got an input has shape of {sar_image.shape}")
        with torch.no_grad():
            netG2 = netG2.to(device)
            startTime = time.time()
            output2 = netG2(sar_image.to(device))
            endTime = time.time()
            logging.info(f"output2: min={output2.min()}, max={output2.max()}")
            logging.info(f"Inference in {endTime - startTime} seconds\n\n")
        # extract contours of inputs and outputs
        try:
            plot_3d(output1.cpu().detach().numpy()[0][0], filename=f"./trained_models/AxTrial-{ax_trial}/{i}-output3d-1{training_typo}_nchan{str(FLAGS.nchan)}.html")
        except:
            logging.debug(f"Error: netG1's plot face an issue")
        plot_3d(output2.cpu().detach().numpy()[0][0], filename=f"./trained_models/AxTrial-{ax_trial}/{i}-output3d-2{training_typo}_nchan{str(FLAGS.nchan)}.html")
        plot_3d(sar_image.cpu().numpy()[0][0], filename=f"./trained_models/AxTrial-{ax_trial}/{i}-input{training_typo}.html")
        plot_3d(label.cpu().numpy()[0][0], filename=f"./trained_models/AxTrial-{ax_trial}/{i}-ideal{training_typo}.html")


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

def get_name(*argv, ax_trial=None):
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
    s += '-depth'+str(FLAGS.depth)+'-'+reduction
    if ax_trial is not None:
        s += '-axtrial'+str(ax_trial)
    return s

def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    return model

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True
    return model