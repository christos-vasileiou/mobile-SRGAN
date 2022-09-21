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
from gan.train_real import *
from gan.eval import compute_prd_from_embedding
import warnings
from guppy import hpy
from tqdm import tqdm
import logging

class TransferLRDataset(Dataset):
    def __init__(self, real_distorted, distorted_data, syn_ideal, transform=None):
        """
        Args:
            distorted_data (np.array): .
            syn_ideal (np.array): .
            transform (callable, optional): Optional transform to be applied
                on a sample. not used
        """
        super(TransferLRDataset, self).__init__()
        self.real_distorted = real_distorted
        self.distorted_data = distorted_data
        self.syn_ideal = syn_ideal

    def __len__(self):
        return len(self.real_distorted)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = np.array(idx.tolist())

        # Assumption:
        # Choose 1 out of 2025 (45 is sqrt of 2025), instead of 2048.
        # Means that samples with index greater that 2025 never used.
        # We have 45 blocks of 45 samples.
        # Generate a random int.
        pos = 45*np.random.randint(0,44) + idx
        distorted_data = self.distorted_data[pos]
        ideal = self.syn_ideal[pos]
        # take the real distorted
        real = self.real_distorted[idx]

        sample = {'real': torch.tensor(real).to(torch.double),
                  'distorted': torch.tensor(distorted_data).to(torch.double),
                  'ideal': torch.tensor(ideal).to(torch.double)
                  }

        return sample

def transfer(netG, netD, real_distorted, test_real_distorted, distorted_data, syn_ideal, n_epochs=100, batch_size=16, lr=0.001, schedFactor=0.1, schedPatience=10,
          weight_decay=0.14, beta1=0.5, workers=1, ngpu=1, device=None):
    # specify the .pt configuration, you load to the Generator.
    warnings.filterwarnings('ignore')
    training_typo, criterion_min_max, criterion_perc, criterion_p2p = get_trained_type(netG)
    if FLAGS.load_model:
        load_G = FLAGS.load_model
    else:
        load_G = './trained_models/Generator' + str(training_typo) + '.pt'
    load_D = './trained_models/Discriminator' + str(training_typo) + '.pt'
    # load the models to train.
    netG.load_state_dict(torch.load(load_G, map_location=device))
    print(load_G)
    #netD.load_state_dict(torch.load(load_D))
    # transfer the models to the available device.
    netG.to(device, dtype=torch.double)
    netD.to(device, dtype=torch.double)

    # freeze first half network (requires_grad = False)
    #netG = freeze(netG)

    # define some variables to keep track of progress.
    epochTimes = []
    G_losses = []
    D_losses = []
    precision = []
    recall = []
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

    for epoch in range(n_epochs):
        dataset = TransferLRDataset(real_distorted=real_distorted, distorted_data=distorted_data, syn_ideal=syn_ideal)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        iterations = len(dataset) // batch_size
        startTime = time.time()
        with tqdm(dataloader, unit='batch') as tepoch:
            for i, batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}:")
                real = batch['real']
                inputs = batch['distorted']
                ideal = batch['ideal']

                ###############################################################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) #
                # (2)              and  minimize L1Loss(fake_xi, real_xi)     #
                ###############################################################

                netD.zero_grad()
                # Format batch
                b_size = ideal.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.double, device=device)
                # Forward pass real batch through D
                real_outputs = netD(ideal.to(device))
                output = real_outputs[-1].view(-1)
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

                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ###############################################
                # (2) Update G network: maximize log(D(G(z))) #
                ###############################################


                #######################################################
                # Pass only the real distorted data through G and D.  #
                # Update G                                            #
                #######################################################
                netG.zero_grad()
                realG = netG(real.to(device)).to(device)
                b_size = real.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.double, device=device)
                # label.fill_(real_label)  # fake labels are real for generator cost
                outputs = netD(realG.detach())
                output = outputs[-1].view(-1)
                errG_real = criterion_min_max(output, label)
                # Calculate gradients for G
                if criterion_min_max.reduction == 'none':
                    for err in errG_real:
                        err.backward(retain_graph=True)
                else:
                    errG_real.backward(retain_graph=True)
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                ################################################
                # GAN's Evaluation: compute_prd_from_embedding #
                ################################################
                eval_data = []
                ref_data = []
                with torch.no_grad():
                    for i in range(len(test_real_distorted)):
                        sar_data = np.expand_dims(test_real_distorted[i], axis=0).astype('float64')
                        #certainty = netD(netG(torch.tensor(sar_data).to(device)))
                        #certainty = certainty[-1].view(-1).mean().item()
                        eval_data.append(netG(torch.tensor(sar_data).to(device)))
                        sar_data = np.expand_dims(syn_ideal[np.random.randint(2025, 2048)], axis=0).astype('float64')
                        #certainty = netD(torch.tensor(sar_data).to(device))
                        #certainty = certainty[-1].view(-1).mean().item()
                        ref_data.append(torch.tensor(sar_data).to(device))
                print(f"eval_data: {eval_data[0].shape}, ref_data: {ref_data[0].shape}")
                prec, rec = compute_prd_from_embedding(eval_data, ref_data, num_clusters=2)
                print(f"prec: {prec.shape} \n{prec}")
                print(f"rec: {rec.shape} \n{rec}")
                precision.append(prec)
                recall.append(rec)

                ################################################

                if D_G_z2 > .9:
                    torch.save(netG.state_dict(), './trained_models/GeneratorReal'+str(training_typo)+'.pt')

                # Output training stats
                epochTimes.append(time.time() - startTime)
                memory_heap = convert_size(hpy().heap().size)
                if criterion_min_max.reduction == 'none':
                    tepoch.set_postfix_str(
                        f'{epochTimes[-1]:.2f}sec: {memory_heap} -> [{epoch}/{n_epochs}][{i}/{len(dataloader)}] -> Loss_D: {errD.sum().item():4.4f}, Loss_G: {errG_real.sum().item():4.4f}, D(x): {D_x:4.4f}, {D_G_z1:4.4f} / {D_G_z2:4.4f} / {certainty:.2f}',
                        refresh=True)
                else:
                    tepoch.set_postfix_str(
                        f'{epochTimes[-1]:.2f}sec: {memory_heap} -> [{epoch}/{n_epochs}][{i}/{len(dataloader)}] -> Loss_D: {errD.item():4.4f}, Loss_G: {errG_real.item():4.4f}, D(x): {D_x:4.4f}, {D_G_z1:4.4f} / {D_G_z2:4.4f} / {certainty:.2f}',
                        refresh=True)

                startTime = time.time()

                # Save Losses for plotting later
                if criterion_min_max.reduction == 'none':
                    G_losses.append(errG_real.sum().item())
                else:
                    G_losses.append(errG_real.item())


    pyplot.figure(figsize=(10, 5))
    pyplot.title("Generator and Discriminator Loss During Training")
    pyplot.plot(G_losses, label="G")
    pyplot.xlabel("iterations")
    pyplot.ylabel("Loss")
    pyplot.legend()
    pyplot.savefig('./trained_models/GLossReal' + str(training_typo) + '.png')

    pyplot.figure(figsize=(10, 5))
    pyplot.title("Generator and Discriminator Precision - Recall")
    pyplot.plot(precision, label="Precision")
    pyplot.plot(recall, label="Recall")
    pyplot.xlabel("iterations")
    pyplot.ylabel("Performance")
    pyplot.legend()
    pyplot.savefig('./trained_models/GPerformance' + str(training_typo) + '.png')
    # Save the model checkpoint
    torch.save(netG.state_dict(), './trained_models/GeneratorReal'+str(training_typo)+'-final.pt')
    print('  ./trained_models/GeneratorReal'+str(training_typo)+'-final.pt file stored. Fully trained model!')
    return results, netG, netD, training_typo

def main(argv):
    warnings.filterwarnings('ignore')
    torch.backends.cudnn.deterministic = True

    # device settings
    torch.set_default_tensor_type('torch.DoubleTensor')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info(f"Device: {bcolors.OKGREEN} {device} {bcolors.ENDC}")
    distorted_data, syn_ideal, train_real_distorted, test_real_distorted = getData(train_pct=0.75, valid_pct=0.125, use_real=True)
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
        nchan = 32
        batch_size = 4
        ndf = 16
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

    results, netG, netD, training_typo = transfer(netG,
                                                  netD,
                                                  train_real_distorted,
                                                  test_real_distorted,
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
    #training_typo = '-Generator-BCELoss-L1Loss-L1Loss-sum'

    run_a_real_sample(None, test_real_distorted, hps, training_typo, device, n_samples=1)

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    app.run(main)