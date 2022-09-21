import numpy as np
import pymatreader
import torch
from absl import flags, app
from absl.flags import FLAGS
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from matplotlib import pyplot as plt
import pandas as pd
from mbd.getData import *
from mbd.Unet import *
from mbd.gan_models import *
from mbd.utils import *
from scipy.ndimage import zoom
import warnings
import logging
from mbd.eval import compute_prd_from_embedding
from torchvision.transforms import transforms
from sklearn.metrics import auc

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
                     'depthwise',
                     'Define if the Generator will be composed of Depth-Wise Conv op in a U-Net shape (`depthwise`) or Pipeline CNN (`pipeline`) or U-Net with normal Conv op (`normal`).')

class EvaluateDataset(Dataset):
    def __init__(self):
        """ Dataset used for evaluation. TestSet
        Args:
        """
        super(EvaluateDataset, self).__init__()
        data = pymatreader.read_mat(f'../../../data/hffh_testing.mat')
        self.radar = np.expand_dims(data['radarImages'],axis=1)
        self.ideal = np.expand_dims(data['idealImages'],axis=1)
        self.length = len(self.radar)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = np.array(idx.tolist())
        radar = torch.tensor(self.radar[idx])
        ideal = torch.tensor(self.ideal[idx])
        sample = {'radar': radar,
                  'ideal': ideal
                  }
        return sample


def train(netG, netD, dataset, n_epochs=100, batch_size=16, lr=0.001, schedFactor=0.1, schedPatience = 10, weight_decay = 0.14, beta1 = 0.5, workers = 1, ngpu = 1, device = None):
    """Train the neural network.
       Args:
           X (RadarDataSet): training data set in the PyTorch data format
           Y (RadarDataSet): test data set in the PyTorch data format
           n_epochs (int): number of epochs to train for
           learning_rate: starting learning rate (to be decreased over epochs by internal scheduler
    """
    warnings.filterwarnings('ignore')

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
    logging.info(f"training type: {training_typo}")
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    # Setup Adam optimizers for both G and D
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay = weight_decay)
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay = weight_decay)
    
    #schedulerG = optim.lr_scheduler.ReduceLROnPlateau(optimizerG, 'min', verbose = True, factor = schedFactor, patience = schedPatience)
    #schedulerD = optim.lr_scheduler.ReduceLROnPlateau(optimizerD, 'min', verbose = True, factor = schedFactor, patience = schedPatience)
    
    # stores results over all epochs
    results = pd.DataFrame(index = list(range(1, n_epochs + 1)),
                            columns = ["Train Loss", "Validation Loss", "Test Loss",
                                       "Min Validation Loss"])

    if FLAGS.pretrained:
        state_dict = torch.load('./trained_models/Generator' + str(training_typo) + '.pt', map_location=device)
        netG.load_state_dict(state_dict)
        #state_dict = torch.load('./trained_models/Discriminator' + str(training_typo) + '.pt', map_location=device)
        #netD.load_state_dict(state_dict)
        logging.info("Generator has been restored!")
        del state_dict
    preprocess = transforms.Resize([256, 256])
    total_time = 0
    iterations = len(dataset) // batch_size
    for epoch in range(n_epochs):
        trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        startTime = time.time()
        for i, data in enumerate(trainloader):
            # get data
            inputs = preprocess(data[:,0])
            ideal = preprocess(data[:,1])
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
            b_size = ideal.size(0)
            ideal = ideal.to(device)
            label = torch.full((b_size,), real_label, dtype=torch.double, device=device)
            # Forward pass real batch through D
            netG = netG.cpu()
            netD = netD.cuda()
            real_outputs = netD(ideal.to(device))
            output = real_outputs[-1].view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion_min_max(output, label)
            real_outputs = [r.detach() for r in real_outputs]
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
            output = fake_outputs[-1].view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion_min_max(output, label)
            fake_outputs = [f.detach() for f in fake_outputs]
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            errD_fake = errD_fake.detach()
            D_G_z1 = output.mean().item()

            # Calculate D's perceptual loss which penalize the discrepancy between intermediate feature maps extracted by D.
            errD_perc = torch.tensor(0.).to(device)
            if criterion_perc:
                netD = netD.cpu()
                netG = netG.cuda()
                fake = netG(inputs).to(device)
                #fake = fake.detach()
                netG = netG.cpu()
                netD = netD.cuda()
                fake_outputs = netD(fake)
                #logging.info(len(real_outputs), len(fake_outputs))
                k=0
                lambda_var = [.001 for _ in range(6)]
                for ro, fo in zip(real_outputs, fake_outputs):
                    k+=1
                    if k == len(real_outputs)-2:
                        break
                    #logging.info(ro.size(), fo.size())
                    errD_perc += criterion_perc(ro, fo)
                fake_outputs = [f.detach() for f in fake_outputs]
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_perc.backward()
                errD_perc = errD_perc.detach()

            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            if criterion_perc:
                errD += errD_perc
            # Update D
            optimizerD.step()

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
            output = fake_outputs[-1].view(-1)
            # Calculate G's loss based on this output
            errG = criterion_min_max(output, label)
            fake_outputs = [f.detach() for f in fake_outputs]
            # Calculate gradients for G
            errG.backward()
            errG = errG.detach()

            errG_p2p = torch.tensor(0.).to(device)
            if criterion_p2p:
                netD = netD.cpu()
                netG = netG.cuda()
                fake = netG(inputs).to(device)
                errG_p2p = criterion_p2p(fake, ideal.to(device))
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errG_p2p.backward()
                errG += errG_p2p
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            netD = netD.cpu()
            netG = netG.cuda()
            g = netG(inputs)
            netG = netG.cpu()
            netD = netD.cuda()
            certainty = netD(g)
            certainty = certainty[-1].view(-1).mean().item()
            psnr = PSNR(y_true=ideal, y_pred=g, MAXp=ideal.max())
            rmse = RMSE(y_true=ideal, y_pred=g)
            if certainty > .99 and g.max() > 5.:
                torch.save(netG.state_dict(), f'./trained_models/{FLAGS.gentype}/{FLAGS.dataset}/Generator'+str(training_typo)+'.pt')
                torch.save(netD.state_dict(), f'./trained_models/{FLAGS.gentype}/{FLAGS.dataset}/Discriminator' + str(training_typo) + '.pt')

            # Output training stats
            if i % 200 == 0 or i==iterations-1:
                epochTimes.append(time.time() - startTime)
                logging.info(f"{epochTimes[-1]:.2f}sec: [{epoch}/{n_epochs}][{i}/{len(trainloader)}] Loss_D: {errD.item():.2f} Loss_D_perc: {errD_perc.item():.2f} Loss_G: {errG.item():.2f} Loss_G_p2p: {errG_p2p.item():.2f} G_max: {g.max().item():.2f} PSNR: {psnr:.2f} RMSE: {rmse:.2f} D(x): {D_x:.2f} D(G(z)): {D_G_z1:.2f}/{D_G_z2:.2f}/{certainty:.2f}")
                startTime = time.time()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        del fake_outputs
        del real_outputs
        del errG
        del errD
        del trainloader

        ################################################
        # GAN's Evaluation: compute_prd_from_embedding #
        ################################################
        evalset = EvaluateDataset()
        evalloader = DataLoader(evalset, batch_size=16, num_workers=workers)
        netD = netD.cpu()
        netG = netG.to(device)
        psnrs = []
        rmses = []
        with torch.no_grad():
            for i, data in enumerate(evalloader):
                sar_data = preprocess(data['radar']).to(device)
                ideal = preprocess(data['ideal']).to(device)
                generated = netG(sar_data).to(device)
                psnr = PSNR(y_true=ideal, y_pred=generated)
                psnrs.append(psnr.cpu().detach().numpy())
                rmse = RMSE(y_true=ideal, y_pred=generated)
                rmses.append(rmse.cpu().detach().numpy())
        psnrs = np.mean(psnrs)
        rmses = np.mean(rmses)
        logging.info(f"PSNR: {bcolors.OKGREEN}{psnrs}{bcolors.ENDC}")
        logging.info(f"RMSE: {bcolors.OKGREEN}{rmses}{bcolors.ENDC}")
        del evalset
        del evalloader
    epochTimes = np.array(epochTimes)
    print_time(epochTimes.sum())

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'./trained_models/{FLAGS.gentype}/{FLAGS.dataset}/GDLoss'+str(training_typo)+'.png')
    plt.close()

    # Save the model checkpoint
    torch.save(netG.state_dict(), f'./trained_models/{FLAGS.gentype}/{FLAGS.dataset}/Generator'+str(training_typo)+'-final.pt')
    torch.save(netD.state_dict(), f'./trained_models/{FLAGS.gentype}/{FLAGS.dataset}/Discriminator'+str(training_typo)+'-final.pt')
    
    return results, netG, netD, training_typo

# OUTDATED!
def evaluate(self, dataset, device, var = None, rmse_eval = False):
    #Evaluate the model 
    with torch.no_grad():
        val_loss = 0
        #correct = 0
        #total = 0
        self.eval()

        bs = len(dataset) if rmse_eval else FLAGS.batch_size
        datasetloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle = True)
        for i, data in enumerate(datasetloader):
            inputs = data[:,0]
            ideal = data[:,1]
            inputs = inputs.to(device)
            #ideal = ideal.to(device)
            
            val_outputs = self(inputs).to(device)
            
            # 10 samples are evaluated by calculating RMSE Loss for comparison among different algorithms.
            if rmse_eval:
                rmse = RMSELoss()
                val_loss = rmse(val_outputs.to(device), ideal.to(device))
                break
            
            # Modify ideal to match the size of the extracted output.
            if FLAGS.retain_shape:
                ideal.to(device)
            else:
                # opencv library using resize() method
                #imgs = [cv2.merge(t) for t in ideal.numpy()]
                #o = val_outputs[0].shape[-1]
                #ideal = np.expand_dims(np.array([cv2.resize(img, dsize=(o, o), interpolation=cv2.INTER_LINEAR) for img in imgs]), axis=1)
            
	        # zoom() method
                hw = val_outputs[0].shape[-1]/ideal.numpy()[0].shape[-1]
                ideal = zoom(ideal[:].numpy(), (1, 1, hw, hw))
                ideal = torch.from_numpy(ideal).to(device)
                        
            # MSE, L1Loss(MAE)
            #val_epoch_loss = self.criterion(val_outputs, ideal)
            
            # Gaussian Negative Loglikelihood Loss
            val_epoch_loss = self.criterion(val_outputs.to(device), ideal.to(device), var.to(device))
            val_loss += val_epoch_loss * bs
    
    if rmse_eval:
        return val_loss
    else:
        return val_loss/len(dataset)


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

    completeDataSet, trainSet, validationSet, testSet = getData(train_pct = 0.75, valid_pct = 0.125)
    try:
        depth = FLAGS.depth
        kernel_size = FLAGS.kernel_size
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
        kernel_size = 3
        nchan = 64
        batch_size = 16
        ndf = 64
        n_epochs = 5
        learning_rate = 0.001
        schedFactor = .1
        schedPatience = 3
        weight_decay = .14
        logging.debug(f"{bcolors.FAIL}An Error was raised by trying to parse FLAGS.{bcolors.ENDC}")
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
        # Beta1 hyperparam for Adam optimizers
        'beta1': 0.5,
        # Number of GPUs available. Use 0 for CPU mode.
        'ngpu': 1
    }

    netG, netD = build_models(hps)
    logging.info(f"\n{netG}\n{netD}")
    G_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    D_params = sum(p.numel() for p in netD.parameters() if p.requires_grad)
    logging.info(f"G {G_params} parameters")
    logging.info(f"D {D_params} parameters")
    _, netG, netD, training_typo = train(netG,
                    netD, 
                    trainSet,
                    n_epochs = n_epochs, 
                    batch_size = batch_size, 
                    lr = learning_rate,
                    schedFactor = schedFactor, 
                    schedPatience = schedPatience, 
                    weight_decay = weight_decay,
                    beta1 = hps['beta1'],
                    workers = hps['workers'],
                    device = device)

    n_samples=10
    run_a_sample(hps, training_typo, n_samples=n_samples)


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    app.run(main)
