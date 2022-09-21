import torch
from absl import flags, app
from absl.flags import FLAGS
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot
import pandas as pd
import numpy as np
import torch.nn as nn
import time, statistics
from gan.getData import *
from gan.Unet import *
from gan.gan_models import *
from gan.utils import *
from scipy.ndimage import zoom
from os.path import exists
from torch.profiler import profile, ProfilerActivity
import warnings
from guppy import hpy
from tqdm import tqdm
import logging

flags.DEFINE_boolean('perc',
                     False,
                     'set if perceptual loss will be applied.')

flags.DEFINE_boolean('p2p',
                     False,
                     'set if pixel-wise loss will be applied between fakes and ideals.')

class RealSynthDataset(Dataset):
    def __init__(self, distorted_data, syn_ideal, transform=None):
        """
        Args:
            distorted_data (np.array): .
            syn_ideal (np.array): .
            transform (callable, optional): Optional transform to be applied
                on a sample. not used
        """
        super(RealSynthDataset, self).__init__()
        self.distorted_data = distorted_data
        self.syn_ideal = syn_ideal

    def __len__(self):
        return len(self.distorted_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = np.array(idx.tolist())
        distorted_data = self.distorted_data[idx]
        if idx > len(self.syn_ideal)-1:
            pos = np.remainder(idx, self.syn_ideal.shape[0])
            ideal = self.syn_ideal[pos]
        else:
            ideal = self.syn_ideal[idx]

        sample = {'distorted': torch.tensor(distorted_data),
                  'ideal': torch.tensor(ideal)}

        return sample


def train(netG, netD, distorted_data, syn_ideal, n_epochs=100, batch_size=16, lr=0.001, schedFactor=0.1, schedPatience=10,
          weight_decay=0.14, beta1=0.5, workers=1, ngpu=1, device=None):
    """Train the neural network.

       Args:
           X (RadarDataSet): training data set in the PyTorch data format
           Y (RadarDataSet): test data set in the PyTorch data format
           n_epochs (int): number of epochs to train for
           learning_rate: starting learning rate (to be decreased over epochs by internal scheduler
    """

    # seed and initialization
    torch.cuda.manual_seed_all(123456)
    torch.manual_seed(123456)

    netG = netG.to(device)
    netD = netD.to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    D_real = []
    D_fake = []
    D_perc = []
    iters = 0
    epochTimes = []

    # set up function loss, optimizer, and learning rate scheduler.
    criterion_p2p = None
    criterion_min_max = None
    criterion_perc = None
    criterion_min_max = nn.BCELoss(reduction='sum').to(device)
    criterion_perc = nn.L1Loss(reduction='sum').to(device) if FLAGS.perc else None
    criterion_p2p = nn.L1Loss(reduction='sum').to(device) if FLAGS.p2p else None
    training_typo = get_name(netG, criterion_min_max, criterion_perc, criterion_p2p)
    logging.info("training type: ", training_typo)
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    # Setup Adam optimizers for both G and D
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay)
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay)

    # schedulerG = optim.lr_scheduler.ReduceLROnPlateau(optimizerG, 'min', verbose = True, factor = schedFactor, patience = schedPatience)
    # schedulerD = optim.lr_scheduler.ReduceLROnPlateau(optimizerD, 'min', verbose = True, factor = schedFactor, patience = schedPatience)

    # stores results over all epochs
    results = pd.DataFrame(index=list(range(1, n_epochs + 1)),
                           columns=["Train Loss", "Validation Loss", "Test Loss",
                                    "Min Validation Loss"])
    # load a model to train, if specified.
    if FLAGS.load_model:
        netG.load_state_dict(torch.load(FLAGS.load_model))

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU]
    with profile(activities=activities, profile_memory=True) as prof:
        for epoch in range(n_epochs+1):
            dataset = RealSynthDataset(distorted_data=distorted_data, syn_ideal=syn_ideal)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
            iterations = len(distorted_data) // batch_size
            startTime = time.time()
            with tqdm(dataloader, unit='batch') as tepoch:
                for i, batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}:")
                    inputs = batch['distorted']
                    ideal = batch['ideal']

                    ###############################################################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) #
                    # (2)              and  minimize L1Loss(fake_xi, real_xi)     #
                    ###############################################################

                    ## Train with all-real batch
                    netD.zero_grad()
                    # Format batch
                    b_size = ideal.size(0)
                    label = torch.full((b_size,), real_label, dtype=torch.double, device=device)
                    # Forward pass real batch through D
                    real_outputs = netD(ideal.to(device))
                    output = real_outputs[-1].view(-1)
                    ideal = prepare_ideal(ideal, output)
                    # Calculate loss on all-real batch
                    errD_real = criterion_min_max(output, label)
                    # Calculate gradients for D in backward pass
                    if criterion_min_max.reduction == 'none':
                        for err in errD_real:
                            err.backward(retain_graph=True)
                    else:
                        errD_real.backward(retain_graph=True)
                    D_x = output.mean().item()

                    ## Train with all-fake batch
                    # Generate fake image batch with G
                    fake = netG(inputs.to(device)).to(device)
                    label.fill_(fake_label)
                    # Classify all fake batch with D
                    fake_outputs = netD(fake.detach())
                    output = fake_outputs[-1].view(-1)
                    # Calculate D's loss on the all-fake batch
                    errD_fake = criterion_min_max(output, label)
                    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                    if criterion_min_max.reduction == 'none':
                        for err in errD_fake:
                            err.backward(retain_graph=True)
                    else:
                        errD_fake.backward(retain_graph=True)
                    D_G_z1 = output.mean().item()

                    # Calculate D's perceptual loss which penalize the discrepancy between intermediate feature maps extracted by D.
                    if criterion_perc:
                        fake_outputs = netD(fake.to(device))
                        # print(len(real_outputs), len(fake_outputs))
                        errD_perc = 0
                        k = 0
                        for ro, fo in zip(real_outputs, fake_outputs):
                            k += 1
                            if k == 7:
                                break
                            # print(ro.size(), fo.size())
                            errD_perc += criterion_perc(ro, fo)
                        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                        if criterion_perc.reduction == 'none':
                            for err in errD_perc:
                                err.backward(retain_graph=True)
                        else:
                            errD_perc.backward(retain_graph=True)

                    # Compute error of D as sum over the fake and the real batches
                    errD = errD_real + errD_fake
                    if criterion_perc:
                        errD += errD_perc
                    # Update D
                    optimizerD.step()

                    ###############################################
                    # (2) Update G network: maximize log(D(G(z))) #
                    ###############################################
                    netG.zero_grad()
                    label.fill_(real_label)  # fake labels are real for generator cost
                    # Since we just updated D, perform another forward pass of all-fake batch through D
                    fake_outputs = netD(fake.to(device))
                    output = fake_outputs[-1].view(-1)
                    # Calculate G's loss based on this output
                    # Adversarial loss + Gaussian Negative Log Likelihood loss fake-real
                    # print(fake.shape, targets.shape)
                    errG = criterion_min_max(output, label)
                    # Calculate gradients for G
                    if criterion_min_max.reduction == 'none':
                        for err in errG:
                            err.backward(retain_graph=True)
                    else:
                        errG.backward(retain_graph=True)

                    if criterion_p2p:
                        errG_p2p = criterion_p2p(fake, ideal.to(device))
                        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                        if criterion_p2p.reduction == 'none':
                            for err in errG_p2p:
                                err.backward(retain_graph=True)
                        else:
                            errG_p2p.backward(retain_graph=True)
                        errG += errG_p2p
                    D_G_z2 = output.mean().item()
                    # Update G
                    optimizerG.step()

                    f = netG(inputs.to(device)).to(device)
                    certainty = netD(f.to(device))
                    certainty = certainty[-1].view(-1).mean().item()

                    if D_G_z2 > .45 and D_G_z2 < .55:
                        torch.save(netG.state_dict(), './trained_models/Generator' + str(training_typo) + '.pt')

                    # Output training stats
                    if i % 20 == 0 or i == iterations - 1:
                        epochTimes.append(time.time() - startTime)
                        memory_heap = convert_size(hpy().heap().size)
                        if criterion_min_max.reduction == 'none':
                            tepoch.set_postfix_str(f'{epochTimes[-1]:.2f}sec: {memory_heap} -> [{epoch}/{n_epochs}][{i}/{len(dataloader)}] -> Loss_D: {errD.sum().item():4.4f}, Loss_G: {errG.sum().item():4.4f}, D(x): {D_x:4.4f}, D(G(z)): {D_G_z1:4.4f} / {D_G_z2:4.4f} / {certainty:.2f}', refresh=True)
                        else:
                            tepoch.set_postfix_str(f'{epochTimes[-1]:.2f}sec: {memory_heap} -> [{epoch}/{n_epochs}][{i}/{len(dataloader)}] -> Loss_D: {errD.item():4.4f}, Loss_G: {errG.item():4.4f}, D(x): {D_x:4.4f}, D(G(z)): {D_G_z1:4.4f} / {D_G_z2:4.4f} / {certainty:.2f}', refresh=True)
                        startTime = time.time()

                    # Save Losses for plotting later
                    if criterion_min_max.reduction == 'none':
                        G_losses.append(errG.sum().item())
                        D_losses.append(errD.sum().item())
                        D_real.append(errD_real.sum().item())
                        D_fake.append(errD_fake.sum().item())
                        if criterion_perc:
                            D_perc.append(errD_perc.sum().item())
                    else:
                        G_losses.append(errG.item())
                        D_losses.append(errD.item())
                        D_real.append(errD_real.item())
                        D_fake.append(errD_fake.item())
                        if criterion_perc:
                            D_perc.append(errD_perc.item())

                    if certainty >= 0.9 and epoch >= n_epochs:
                        break

    prof.export_chrome_trace("./trace"+ str(training_typo) +".json")
    if torch.cuda.is_available():
        print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    else:
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    pyplot.figure(figsize=(10, 5))
    pyplot.title("Generator and Discriminator Loss During Training")
    pyplot.plot(G_losses, label="G")
    pyplot.plot(D_losses, label="D")
    pyplot.plot(D_real, label="D real")
    pyplot.plot(D_fake, label="D fake")
    if criterion_perc:
        pyplot.plot(D_perc, label="D perc")

    pyplot.xlabel("iterations")
    pyplot.ylabel("Loss")
    pyplot.legend()
    pyplot.savefig('./trained_models/GDLoss' + str(training_typo) + '.png')
    # Save the model checkpoint
    torch.save(netG.state_dict(), './trained_models/Generator' + str(training_typo) + '-final.pt')

    return results, netG, netD, training_typo


# OUTDATED!
def evaluate(self, dataset, device, var=None, rmse_eval=False):
    # Evaluate the model
    with torch.no_grad():
        val_loss = 0
        # correct = 0
        # total = 0
        self.eval()

        bs = len(dataset) if rmse_eval else FLAGS.batch_size
        datasetloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
        for i, data in enumerate(datasetloader):
            inputs = data[:, 0]
            targets = data[:, 1]
            inputs = inputs.to(device)
            # targets = targets.to(device)

            val_outputs = self(inputs).to(device)

            # 10 samples are evaluated by calculating RMSE Loss for comparison among different algorithms.
            if rmse_eval:
                rmse = RMSELoss()
                val_loss = rmse(val_outputs.to(device), targets.to(device))
                break

            # Modify Targets to match the size of the extracted output.
            if FLAGS.retain_shape:
                targets.to(device)
            else:
                # opencv library using resize() method
                # imgs = [cv2.merge(t) for t in targets.numpy()]
                # o = val_outputs[0].shape[-1]
                # targets = np.expand_dims(np.array([cv2.resize(img, dsize=(o, o), interpolation=cv2.INTER_LINEAR) for img in imgs]), axis=1)

                # zoom() method
                hw = val_outputs[0].shape[-1] / targets.numpy()[0].shape[-1]
                targets = zoom(targets[:].numpy(), (1, 1, hw, hw))
                targets = torch.from_numpy(targets).to(device)

            # MSE, L1Loss(MAE)
            # val_epoch_loss = self.criterion(val_outputs, targets)

            # Gaussian Negative Loglikelihood Loss
            val_epoch_loss = self.criterion(val_outputs.to(device), targets.to(device), var.to(device))
            val_loss += val_epoch_loss * bs

    if rmse_eval:
        return val_loss
    else:
        return val_loss / len(dataset)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

def main(argv):
    warnings.filterwarnings('ignore')
    torch.backends.cudnn.deterministic = True

    # device settings
    torch.set_default_tensor_type('torch.DoubleTensor')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    distorted_data, syn_ideal, real_distorted = getData(train_pct=0.75, valid_pct=0.125, use_real=True)
    try:
        depth = FLAGS.depth
        nchan = FLAGS.nchan
        batch_size = FLAGS.batch_size
        ndf = FLAGS.ndf
        n_epochs = FLAGS.n_epochs
        learning_rate = FLAGS.learning_rate
        schedFactor = FLAGS.schedFactor
        schedPatience = FLAGS.schedPatience
        weight_decay = FLAGS.weight_decay
    except:
        depth = 2
        nchan = 64
        batch_size = 16
        ndf = 64
        n_epochs = 5
        learning_rate = 0.001
        schedFactor = .1
        schedPatience = 3
        weight_decay = .14
        print("An Error was raised by trying to parse FLAGS.")
    hps = {
        'depth': depth,
        'nchan': nchan,
        # batch size during training
        'batch_size': batch_size,
        # Number of workers for dataloader
        'workers': 4,
        # Number of channels in the training images.
        'nc': 1,
        # Size of feature maps in discriminator
        'ndf': ndf,
        # Beta1 hyperparam for Adam optimizers
        'beta1': 0.5,
        # Number of GPUs available. Use 0 for CPU mode.
        'ngpu': 1
    }

    netG = Generator(hps, retain_shape_to_output=FLAGS.retain_shape)
    netD = Discriminator(hps)

    print(netG)
    print(netD)
    _, netG, netD, training_typo = train(netG,
                                         netD,
                                         distorted_data,
                                         syn_ideal,
                                         n_epochs=n_epochs,
                                         batch_size=batch_size,
                                         lr=learning_rate,
                                         schedFactor=schedFactor,
                                         schedPatience=schedPatience,
                                         weight_decay=weight_decay,
                                         beta1=hps['beta1'],
                                         workers=hps['workers'],
                                         ngpu=hps['ngpu'],
                                         device=device)

    run_a_sample(None, real_distorted, hps, training_typo, device)

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    app.run(main)
