import pymatreader
import torch
from absl import flags, app
from absl.flags import FLAGS
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
from axdata3d.get3data import *
from axdata3d.Unet import *
from axdata3d.gan_models import *
from axdata3d.utils import *
from axdata3d.dataset import FeedDataset3d
import warnings
import os
import logging
from os.path import exists
from torchvision import transforms
from torchinfo import summary
import wandb

flags.DEFINE_boolean('perc',
                     False,
                     'set if perceptual loss will be applied.')

flags.DEFINE_boolean('p2p',
                     False,
                     'set if pixel-wise loss will be applied between fakes and ideals.')

flags.DEFINE_boolean('pretrained',
                     False,
                     'Load pretrained model to keep training it.')

flags.DEFINE_string('gentype',
                     'normal',
                     'Define if the convolutional operation will be normal/depthwise/pipeline.')

flags.DEFINE_integer('doutput_size',
                     4,
                     'Size of patches for Discriminator`s output. if doutput_size is 4, then the Dscriminator output`s size is 4x4x4')

flags.DEFINE_integer('input_size',
                     64,
                     'Generator`s input size')

flags.DEFINE_integer('dataset_size',
                     None,
                     'Size of the dataset you are to use for training and testing.')

flags.DEFINE_float('split',
                   0.8,
                   'set the factor of the scheduler {float}.')

def train(netG, netD, hps, ax_trial='Train-3d'):
    """Train the neural network.

       Args:
           X (RadarDataSet): training data set in the PyTorch data format
           Y (RadarDataSet): test data set in the PyTorch data format
           n_epochs (int): number of epochs to train for
           learning_rate: starting learning rate (to be decreased over epochs by internal scheduler
    """
    if not exists(f'./trained_models/AxTrial-{ax_trial}'):
        os.mkdir(f'./trained_models/AxTrial-{ax_trial}')
    # definition and initialization
    min_rmse = np.inf
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    n_epochs = hps['n_epochs']
    batch_size = hps['batch_size']
    lrG = hps['learning_rateG']
    lrD = hps['learning_rateG']
    weight_decayG = hps['weight_decayG']
    weight_decayD = hps['weight_decayD']
    beta1 = hps['beta1']
    workers = hps['workers']
    ngpu = hps['ngpu']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pixel_lambda = hps['pixel_lambda']
    perc_lambda = [hps[f"perc_lambda_{i}"] for i in range(1,6)] # lambdas for each layer of the Discriminator

    # seed and initialization
    torch.cuda.manual_seed_all(123456)
    torch.manual_seed(123456)

    # Handle multi-gpu if desired
    netG = netG.to(device)
    netD = netD.to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    epochTimes = []

    # set up function loss, optimizer, and learning rate scheduler.
    criterion_min_max = nn.BCELoss(reduction='sum').to(device)
    criterion_perc = nn.MSELoss(reduction='sum').to(device) if FLAGS.perc else None
    criterion_p2p = nn.MSELoss(reduction='sum').to(device) if FLAGS.p2p else None
    training_typo = get_name(netG, criterion_min_max, criterion_perc, criterion_p2p)
    logging.info(f"training type: {str(training_typo)}")
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    # Setup Adam optimizers for both G and D
    optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999), weight_decay=weight_decayG)
    optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999), weight_decay=weight_decayD)

    # stores results over all epochs
    results = pd.DataFrame(index=list(range(1, n_epochs + 1)),
                           columns=["RMSE", "PSNR"])
    # load a model to train, if specified.
    if FLAGS.pretrained:
        state_dict = torch.load(f'./trained_models/AxTrial-{ax_trial}/Generator' + str(training_typo) + '.pt', map_location=device)
        netG.load_state_dict(state_dict)
        state_dict = torch.load(f'./trained_models/AxTrial-{ax_trial}/Discriminator' + str(training_typo) + '.pt', map_location=device)
        netD.load_state_dict(state_dict)
        logging.info("Generator and Discriminator has been restored!")
        del state_dict

    #wandb.watch(netG, log="all", log_freq=5)
    #wandb.watch(netD, log="all", log_freq=5)
    for epoch in range(n_epochs):
        dataset = FeedDataset3d(mode=Mode.training, split=FLAGS.split, dataset_size=FLAGS.dataset_size)
        trainloader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)
        if epoch == 0:
            logging.info(f"Samples of dataset: {len(trainloader)}")
        iterations = len(trainloader)
        torch.cuda.empty_cache()
        startTime = time.time()
        for i, data in enumerate(trainloader):
            # get data
            inputs = data['data3d'].to(device)
            ideal = data['ideal'].to(device)
            if i==0 and epoch==0:
                logging.info(f"inputs shape: {inputs.shape}, ideal shape: {ideal.shape}")
            ###############################################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) #
            # (2)              and  minimize L1Loss(fake_xi, real_xi)     #
            ###############################################################

            netG = freeze(netG) # to avoid computation
            netD = unfreeze(netD)

            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            # b_size = ideal.size(0)
            # Forward pass real batch through D
            netG = netG.cpu()
            netD = netD.cuda()
            #real_outputs = run_handler(netD, ideal)
            real_outputs = netD(ideal)
            output = real_outputs[-1]
            label = torch.full((output.shape[0]*output.shape[-1]*output.shape[-2]*output.shape[-3],), real_label, dtype=torch.float32, device=device)
            output = output.view(-1).to(device)
            #logging.info(f"real_outputs: {bcolors.OKGREEN}{output.shape}|{label.shape}{bcolors.ENDC}")
            # Calculate loss on all-real batch
            errD_real = criterion_min_max(output, label)
            real_outputs = [r.detach() for r in real_outputs]
            #logging.info(f"errorD real: {bcolors.OKGREEN}{errD_real}{bcolors.ENDC}")
            # Calculate gradients for D in backward pass
            errD_real.backward()
            errD_real = errD_real.detach()
            D_x = output.mean().item()

            ## Train with all-fake batch
            inputs = inputs.to(device)
            # Generate fake image batch with G
            netD = netD.cpu()
            netG = netG.cuda()
            fake = netG(inputs).to(device)
            fake = fake.detach()
            label.fill_(fake_label)
            # Classify all fake batch with D
            netG = netG.cpu()
            netD = netD.cuda()
            fake_outputs = netD(fake)
            output = fake_outputs[-1].view(-1).to(device)
            #logging.info(f"fake after D: {bcolors.OKGREEN}{output.shape}{bcolors.ENDC}")
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion_min_max(output, label)
            fake_outputs = [f.detach() for f in fake_outputs]
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            errD_fake = errD_fake.detach()
            #torch.cuda.synchronize()
            D_G_z1 = output.mean().item()

            # Calculate D's perceptual loss which penalize the discrepancy between intermediate feature maps extracted by D.
            errD_perc = torch.tensor(0.).to(device)
            if criterion_perc:
                netD = netD.cpu()
                netG = netG.cuda()
                fake = netG(inputs).to(device)
                netD = netD.cuda()
                fake_outputs = netD(fake)
                # logging.info(len(real_outputs), len(fake_outputs))
                k = 0
                lambda_var = [perc_lambda[i] for i in range(len(real_outputs)-2)] # len(real_outputs) = layers of Discriminator = 5
                for ro, fo, lv in zip(real_outputs, fake_outputs, lambda_var):
                    k += 1
                    if i == 0 and epoch == 0 and k==0:
                        logging.info(f"real_outputs length: {len(real_outputs)}")
                    if k == len(real_outputs)-2:
                        break
                    # logging.info(ro.size(), fo.size())
                    errD_perc += criterion_perc(ro, fo) * lv
                fake_outputs = [f.detach() for f in fake_outputs]
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_perc.backward()
                errD_perc = errD_perc.detach()
            #torch.cuda.synchronize()

            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            if criterion_perc:
                errD += errD_perc
            # Update D
            optimizerD.step()
            # clean
            del errD_real
            del errD_fake

            ###############################################
            # (2) Update G network: maximize log(D(G(z))) #
            ###############################################
            netD = freeze(netD) # to avoid computation
            netG = unfreeze(netG)

            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            netD = netD.cpu()
            netG = netG.cuda()
            fake = netG(inputs).to(device)
            netD = netD.cuda()
            fake_outputs = netD(fake)
            output = fake_outputs[-1].view(-1).to(device)
            # Calculate G's loss based on this output
            # Adversarial loss + Gaussian Negative Log Likelihood loss fake-real
            # logging.info(fake.shape, ideal.shape)
            errG = criterion_min_max(output, label) * 1.
            fake_outputs = [f.detach() for f in fake_outputs]
            # Calculate gradients for G
            errG.backward()
            errG = errG.detach()
            #torch.cuda.synchronize()

            errG_p2p = torch.tensor(0.).to(device)
            if criterion_p2p:
                netD = netD.cpu()
                netG = netG.cuda()
                fake = netG(inputs)
                fake = fake.to(device)
                errG_p2p = criterion_p2p(fake, ideal.to(device)) * pixel_lambda
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errG_p2p.backward()
                errG_p2p = errG_p2p.detach()
                errG += errG_p2p
            #torch.cuda.synchronize()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            netD = netD.cpu()
            netG = netG.cuda()
            g = netG(inputs)

            # Normalize
            #z = torch.concat([inputs, ideal, g])
            #z = normalize(z)
            #inputs = z[:z.shape[0] // 3]
            #ideal = z[z.shape[0] // 3: 2 * (z.shape[0] // 3)]
            #g = z[2 * (z.shape[0] // 3):]

            psnr = torch.mean(PSNR(y_true=ideal, y_pred=g))
            rmse = torch.mean(RMSE(y_true=ideal, y_pred=g))
            wandb.log({'RMSE': rmse.item()
                       })

            if g.max() > ideal.max()//2 and rmse.item() < min_rmse:
                torch.save(netG.cpu().state_dict(), f'./trained_models/AxTrial-{ax_trial}/Generator3d{str(training_typo)}.pt')
                torch.save(netD.cpu().state_dict(), f'./trained_models/AxTrial-{ax_trial}/Discriminator3d{str(training_typo)}.pt')

            # Output training stats
            if i % 100 == 0 or i == iterations - 1:
                epochTimes.append(time.time() - startTime)
                logging.info(f"{epochTimes[-1]:.2f}sec: [{epoch}/{n_epochs}][{i}/{len(trainloader)}] Loss_D: {errD.item():.2f} Loss_D_perc: {errD_perc.item():.2f} Loss_G: {errG.item():.2f} Loss_G_p2p: {errG_p2p.item():.2f} G_max: {g.max().item():.3f} PSNR: {psnr:.2f} RMSE: {rmse:.2f} D(x): {D_x:.2f} D(G(z)): {D_G_z1:.2f}/{D_G_z2:.2f}")
                startTime = time.time()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        del errD, errD_perc # , g, z, psnr, rmse, real_outputs, fake_outputs,  errG, errG_p2p, dataset, trainloader

        ################################################
        # GAN's Evaluation: Calculating PSNR and RMSE  #
        ################################################
        evalset = FeedDataset3d(mode=Mode.testing, split=FLAGS.split,  dataset_size=FLAGS.dataset_size)
        evalloader = DataLoader(evalset, batch_size=16, num_workers=workers)
        netD = netD.cpu()
        netG = netG.to(device)
        psnrs = torch.tensor([])
        psnrs_inputs = torch.tensor([])
        rmses = torch.tensor([])
        rmses_inputs = torch.tensor([])
        torch.cuda.empty_cache()
        with torch.no_grad():
            for i, data in enumerate(evalloader):
                inputs = data["data3d"].to(device)
                ideal = data["ideal"].to(device)
                generated = netG(inputs).to(device)

                #z = torch.concat([inputs, ideal, generated])
                #z = normalize(z)
                #inputs = z[:z.shape[0] // 3]
                #ideal = z[z.shape[0] // 3: 2 * (z.shape[0] // 3)]
                #generated = z[2 * (z.shape[0] // 3):]

                # Output's Evaluation
                psnr = PSNR(y_true=ideal, y_pred=generated)
                psnrs = torch.concat([psnrs, psnr.cpu().detach()])
                rmse = RMSE(y_true=ideal, y_pred=generated)
                rmses = torch.concat([rmses, rmse.cpu().detach()])

                psnr = PSNR(y_true=ideal, y_pred=inputs)
                psnrs_inputs = torch.concat([psnrs_inputs, psnr.cpu().detach()])
                rmse = RMSE(y_true=ideal, y_pred=inputs)
                rmses_inputs = torch.concat([rmses_inputs, rmse.cpu().detach()])

        psnrs = torch.mean(psnrs).item()
        rmses = torch.mean(rmses).item()
        wandb.log({'Eval RMSE': rmses
                   })

        results["RMSE"][epoch+1] = rmses
        results["PSNR"][epoch+1] = psnrs
        logging.info(f"Output's PSNR: {bcolors.OKGREEN}{psnrs}{bcolors.ENDC} RMSE: {bcolors.OKGREEN}{rmses}{bcolors.ENDC}")
        logging.info(f"Input's PSNR: {bcolors.OKGREEN}{psnrs}{bcolors.ENDC} RMSE: {bcolors.OKGREEN}{rmses}{bcolors.ENDC}")
        del evalset, evalloader #, psnr, psnrs, psnrs_inputs, rmse, rmses, rmses_inputs, generated, z, ideal, inputs

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'./trained_models/AxTrial-{ax_trial}/GDLoss{str(training_typo)}.png')
    # Save the model checkpoint
    torch.save(netG.cpu().state_dict(), f'./trained_models/AxTrial-{ax_trial}/Generator3d{str(training_typo)}-final.pt')
    torch.save(netD.cpu().state_dict(), f'./trained_models/AxTrial-{ax_trial}/Discriminator3d{str(training_typo)}-final.pt')
    return results, netG, netD, training_typo

def main(argv):
    warnings.filterwarnings('ignore')
    torch.backends.cudnn.deterministic = True

    # device settings
    #logging.info(f"\n{show_install()}")
    torch.set_default_tensor_type('torch.DoubleTensor')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Number of GPUs in the system: {torch.cuda.device_count()}")
    logging.info(torch.cuda.get_device_properties(device=torch.device("cuda")))
    logging.info(f"Device: {device}")

    hps = hps_initialization()
    netG, netD = build_models(hps)
    netG = netG.float()
    netD = netD.float()

    logging.info(netG)
    logging.info(netD)
    G_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    D_params = sum(p.numel() for p in netD.parameters() if p.requires_grad)

    logging.info(f"G {G_params} parameters")
    logging.info(f"D {D_params} parameters")

    wandb.init(project="tiftrc-3d", name='Train-3d', entity="chrivasileiou", config=hps)
    _, netG, netD, training_typo = train(netG,
                                        netD,
                                        hps=hps,
                                        )
    run_a_3d_sample(netG2=None, features=None, labels=None, hps=hps, training_typo=training_typo, device=device, n_samples=10)
    wandb.finish()



if __name__ == "__main__":
    FLAGS = flags.FLAGS
    app.run(main)
