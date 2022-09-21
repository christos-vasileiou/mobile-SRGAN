from absl import flags, app
from absl.flags import FLAGS
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from matplotlib import pyplot
import pandas as pd
import numpy as np
import torch.nn as nn
import time, statistics
from solidlib.getData import getData
from solidlib.Unet import *
from scipy.ndimage import zoom
import cv2

def train(self, trainSet, validationSet, testSet, n_epochs=100, batch_size=16, lr=0.001, schedFactor=0.1, schedPatience = 10, weight_decay = 0., device = None):
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

    epochTimes = []
    my_images = []

    # set up function loss, optimizer, and learning rate scheduler.
    
    #self.criterion = nn.L1Loss()
    #self.criterion = nn.MSELoss()
    self.criterion = nn.GaussianNLLLoss()
    self.criterion = self.criterion.to(device)
    optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose = True, factor = schedFactor, patience = schedPatience)
    
    # stores results over all epochs
    results = pd.DataFrame(index = list(range(1, n_epochs + 1)), 
                            columns = ["Train Loss", "Validation Loss", "Test Loss",
                                       "Min Validation Loss"])
    # load a model to train, if specified.
    if FLAGS.load_model:
        self.load_state_dict(torch.load(FLAGS.load_model))

    for epoch in range(n_epochs):
        startTime = time.time()
        trainloader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle = True)
        iterations = len(trainSet)//batch_size
        
        for i, data in enumerate(trainloader):
            # get data
            #print(f"data : {data.shape}")
            inputs = data[:,0]
            targets = data[:,1]
            inputs = inputs.to(device)
            
            if epoch == 0 and i == 0:
                var = None
	        #var = torch.ones(inputs.shape, requires_grad=True) # heteroscedastic
                var = torch.ones((inputs.shape[0], 1), requires_grad=True).to(device) # homoscedastic
                #print(f"var: {var.shape}")
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = self(inputs).to(device)
            
            # convert targets when output size could have been resized due to resolution reduction of Unet.
            if FLAGS.retain_shape:
                targets.to(device)
            else:
                #imgs = [cv2.merge(t) for t in targets.numpy()]
                #o = outputs[0].shape[-1]
                #targets = np.expand_dims(np.array([cv2.resize(img, dsize=(o, o), interpolation=cv2.INTER_LINEAR) for img in imgs]), axis=1)
                # zoom() method
                hw = outputs[0].shape[-1]/targets.numpy()[0].shape[-1]
                targets = zoom(targets[:].numpy(), (1, 1, hw, hw))
                #print(f"inputs: {inputs.shape}, targets: {targets.shape}, outputs: {outputs.shape}") 
                targets = torch.from_numpy(targets).to(device)
            # MSE, L1Loss(MAE)
            #loss = self.criterion(outputs, targets)
            
            # Gaussian Negative Loglikelihood Loss
            loss = self.criterion(outputs, targets, var)
            
            loss.backward()
            optimizer.step()
            #if (i)%batch_size == 0:
            #    print(f"Epoch [{epoch}/{n_epochs}], Step [{i}/{iterations}], Loss: {loss.item()}")
       
        #model.train()
        train_loss = evaluate(self, trainSet, device, var)
        val_loss = evaluate(self, validationSet, device, var)
        test_loss = evaluate(self, testSet, device, var)

        # lower the learning rate if the validation loss stagnates
        scheduler.step(val_loss)

        # store results
        results.loc[epoch + 1]["Train Loss"] = train_loss 
        results.loc[epoch + 1]["Test Loss"] = test_loss
        results.loc[epoch + 1]["Validation Loss"] = val_loss
        results.loc[epoch + 1]["Min Validation Loss"] = results["Validation Loss"].min()
        epochTimes.append(time.time() - startTime)
        print(f"> {epoch+1}/{n_epochs}, {epochTimes[epoch]:.5f}sec, train loss: {train_loss:.5f}, validation loss: {val_loss:.5f}, test loss: {test_loss:.5f}")
        
    # Save the model checkpoint
    #torch.save(self.state_dict(), 'DenoiseNet.pt')
    plotResults(results)
    print(results.tail(1))
    print(f"Average {round(statistics.mean(epochTimes),2)}s/epoch")
    return results, self


def evaluate(self, dataset, device, var = None):
    #Evaluate the model 
    with torch.no_grad():
        val_loss = 0
        #correct = 0
        #total = 0
        self.eval()

        bs = FLAGS.batch_size
        datasetloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle = True)
        for i, data in enumerate(datasetloader):
            inputs = data[:,0]
            targets = data[:,1]
            inputs = inputs.to(device)
            #targets = targets.to(device)
            
            val_outputs = self(inputs).to(device)
            
            # Modify Targets to match the size of the extracted output.
            if FLAGS.retain_shape:
                targets.to(device)
            else:
                # opencv library using resize() method
                #imgs = [cv2.merge(t) for t in targets.numpy()]
                #o = val_outputs[0].shape[-1]
                #targets = np.expand_dims(np.array([cv2.resize(img, dsize=(o, o), interpolation=cv2.INTER_LINEAR) for img in imgs]), axis=1)
            
	        # zoom() method
                hw = val_outputs[0].shape[-1]/targets.numpy()[0].shape[-1]
                targets = zoom(targets[:].numpy(), (1, 1, hw, hw))
                targets = torch.from_numpy(targets).to(device)
            #if i == 0:
            #    pyplot.figure()
            #    pyplot.contour(targets[0][0])
            #    pyplot.title("Modified Ideal Data")
            #    pyplot.savefig(f"cv2-modified-Ideal-data-{targets.shape[-1]}")
            #    pyplot.close('all')
            
            # MSE, L1Loss(MAE)
            #val_epoch_loss = self.criterion(val_outputs, targets)
            
            # Gaussian Negative Loglikelihood Loss
            val_epoch_loss = self.criterion(val_outputs, targets, var)

            val_loss += val_epoch_loss * bs
    return val_loss/len(dataset)


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


def run_a_sample(model, dataset, device):
    #model = DenoiseNet(dataset[0].shape[1])
    #state_dict = torch.load(modelPath, map_location=device)
    #model.load_state_dict(state_dict)
    
    model.to(device)

    # dataset[m][n]: 
    # --- m -> sample of a dataset, 
    # --- n -> 0: SAR image, 1:Ideal image
    print(f"input: {dataset[0][0].shape}")
    sar_image = np.expand_dims(dataset[0][0], axis=0)
    output = model(torch.from_numpy(sar_image).to(device)).to(device)
    #print(f"output: {output.shape}, output: {output.detach().numpy().shape}, {type(output.detach().numpy())}")
    #print(f"output: {output.detach().numpy()}\n")
    
    # extract contours of inputs and outputs.
    pyplot.figure()
    pyplot.contour((dataset[0][0][0]))
    pyplot.title (f"Radar Data")
    pyplot.savefig(f"input.jpg")
    
    #print(type(output), output.shape, '\n', output)
    pyplot.figure()
    pyplot.contour((output.cpu().detach().numpy()[0][0]))
    pyplot.title (f"Denoised Data")
    pyplot.savefig(f"unet_output_"+FLAGS.o_name+f"_depth{str(FLAGS.depth)}_nchan{str(FLAGS.nchan)}.png")
    pyplot.close('all')
    
def main(argv):
    del argv
    torch.backends.cudnn.deterministic = True
    
    #device settings
    torch.set_default_tensor_type('torch.DoubleTensor')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    completeDataSet, trainSet, validationSet, testSet = getData(train_pct = 0.75, valid_pct = 0.125)
    hps = {
        'depth': FLAGS.depth,
        'nchan': FLAGS.nchan,
        'batch_size': FLAGS.batch_size
    }

    model = Unet(hps, retain_shape_to_output=FLAGS.retain_shape)
    print(model)
    model.to(device) # move network to GPU if present
    results, _ = train(model, 
                    trainSet, validationSet, testSet, 
                    n_epochs = FLAGS.n_epochs, 
                    batch_size = FLAGS.batch_size, 
                    lr = FLAGS.learning_rate, 
                    schedFactor = FLAGS.schedFactor, 
                    schedPatience = FLAGS.schedPatience, 
                    weight_decay = FLAGS.weight_decay,
                    device = device);
    
    run_a_sample(model, validationSet, device)

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    app.run(main)
