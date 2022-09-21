from absl import flags, app
from absl.flags import FLAGS
from ax.service.ax_client import AxClient
from solidlib.getData import getData
from solidlib.Unet import *
from solidlib.train import *
from matplotlib import pyplot
from pandas.plotting import table
import numpy
import time, statistics
from scipy.ndimage import zoom
import cv2

def run_a_sample_ax(model, validset, testset, device, ax_trial):
    # dataset[m][n]: 
    # --- m -> sample of a dataset, 
    # --- n -> 0: SAR image, 1:Ideal image
    print(f"input: {validset[0][0].shape}")
    sar_image = np.expand_dims(validset[0][0], axis=0)
    output = model(torch.from_numpy(sar_image).to(device)).to(device)
    #print(f"output: {output.shape}, output: {output.detach().numpy().shape}, {type(output.detach().numpy())}")
    #print(f"output: {output.detach().numpy()}\n")
    print(type(validset[0]))
    # validation set: extract contours of inputs, outputs, target and modified target.
    pyplot.figure()
    pyplot.contour((validset[0][1][0]))
    pyplot.title (f"Ideal Data")
    pyplot.savefig(f"target-valid.png")
    pyplot.close()
    
    # Modify Target if needed
    if FLAGS.retain_shape:
        target = validset[0]
        print(f"run a sample ax target size: {target[1].shape}")
    else:    
        # opencv using resize() method
        #target = cv2.merge(validset[0][1])
        #o = output[0].shape[-1]
        #target = np.array(cv2.resize(target, dsize=(o,o), interpolation=cv2.INTER_LINEAR))
        # zoom() method
        hw = output[0].shape[-1]/validset[0][1].shape[-1]
        print(f"run a sample ax target size: {validset[0][1].shape}")
        target = np.array(zoom(validset[0], (1,1,hw,hw), order=1))
    
        pyplot.figure()
        print(target.shape)
        pyplot.contour((target[1][0]))
        pyplot.title (f"Modified Ideal Data")
        pyplot.savefig(f"modified-target-valid.png")
        pyplot.close()
    print(f"run a sample ax input size: {target[0].shape}")
    
    pyplot.figure()
    pyplot.contour((validset[0][0][0]))
    pyplot.title (f"SAR Data")
    pyplot.savefig(f"input-valid.png")
    pyplot.close()
    
    pyplot.figure()
    pyplot.contour((output.cpu().detach().numpy()[0][0]))
    pyplot.title (f"Denoised Data")
    pyplot.savefig(ax_trial+"-"+FLAGS.o_name+"-valid.png")
    pyplot.close()
    
    
    sar_image = np.expand_dims(testset[0][0], axis=0)
    output = model(torch.from_numpy(sar_image).to(device)).to(device)
    #print(f"output: {output.shape}, output: {output.detach().numpy().shape}, {type(output.detach().numpy())}")
    #print(f"output: {output.detach().numpy()}\n")
    
    # TEST SET: extract contours of inputs and outputs.
    pyplot.figure()
    pyplot.contour((testset[0][1][0]))
    pyplot.title (f"Ideal Data")
    pyplot.savefig(f"target-test.png")
    pyplot.close()
    
    # Modify Target if needed
    if FLAGS.retain_shape: 
        pass    # do nothing. There is no need to print out.
    else:
        # opencv using resize() method
        #target = cv2.merge(testset[0][1])
        #o = output[0].shape[-1]
        #target = np.array(cv2.resize(target, dsize=(o,o), interpolation=cv2.INTER_LINEAR))
        
        # zoom() method
        hw = output[0].shape[-1]/testset[0][1].shape[-1]
        target = np.array(zoom(testset[0], (1,1,hw,hw), order=1))    
        pyplot.figure()
        pyplot.contour((target[1][0]))
        pyplot.title (f"Modified Ideal Data")
        pyplot.savefig(f"modified-target-test.png")
        pyplot.close()
    
    pyplot.figure()
    pyplot.contour((testset[0][0][0]))
    pyplot.title (f"SAR Data")
    pyplot.savefig(f"input-test.png")
    pyplot.close()
    
    pyplot.figure()
    pyplot.contour((output.cpu().detach().numpy()[0][0]))
    pyplot.title (f"Denoised Data")
    pyplot.savefig(ax_trial+"-"+FLAGS.o_name+"-test.png")
    pyplot.close()

def main(argv):
    del argv
    hps = {
        'batch_size': FLAGS.batch_size,
        'nchan': FLAGS.nchan,
        'lr': FLAGS.learning_rate,
        'weight_decay': FLAGS.weight_decay,
        'depth': FLAGS.depth,
        'momentum': 0.3,
        'schedFactor': FLAGS.schedFactor,
        'schedPatience': FLAGS.schedPatience,
        'n_epochs': FLAGS.n_epochs,
        'optimizer': 'Adam',
        'criterion': 'GaussianNLLLoss',
        'trial_time': 0
    }
    
    ax_client = AxClient()
    print(type(ax_client))
    ax_client.create_experiment(
        name="optimize_hps",
        parameters=[
            {
             "name": "lr",
             "type": "range",
             "bounds": [1e-6, 1e-1],
             "log_scale": True
            },
            {
             "name": "schedFactor",
             "type": "range",
             "bounds": [0.01, 0.2],
             "log_scale": True
            },
            {
             "name": "schedPatience",
             "type": "range",
             "bounds": [2, 4],
             "log_scale": True
            },
            {
             "name": "weight_decay",
             "type": "range",
             "bounds": [1e-2, 1e-0],
             "log_scale": True
            },
            {
             "name": "depth",
             "type": "range",
             "bounds": [2, 3],
             "log_scale": True
            },
            {
             "name": "nchan",
             "type": "range",
             "bounds": [4, 5],
             "log_scale": True
            },
        ],
        #Booth function
        objective_name = 'validation',
        minimize = True,
    )
    torch.backends.cudnn.deterministic = True
    
    #device settings
    torch.set_default_tensor_type('torch.DoubleTensor')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    completeDataSet, trainSet, validationSet, testSet = getData(train_pct = 0.75, valid_pct = 0.125)
    
    for i in range(10):
        
        startT = time.time()
        parameters, trial_index = ax_client.get_next_trial()
        print(f"parameters: {type(parameters)}\n{parameters}")
        hps.update(parameters)
        model = Unet(hps, printForward = False, retain_shape_to_output = FLAGS.retain_shape)
        print(model)
        #exit()
        model.to(device) # move network to GPU if present
        results, model = train(model, 
                    trainSet, validationSet, testSet, 
                    n_epochs = hps['n_epochs'], 
                    batch_size = hps['batch_size'], 
                    lr = hps['lr'], 
                    schedFactor = hps['schedFactor'], 
                    schedPatience = hps['schedPatience'], 
                    weight_decay = hps['weight_decay'],
                    device = device);

        train_loss, valid_loss, test_loss = results["Train Loss"], results["Validation Loss"], results["Test Loss"]
        
        raw_data = {
            'train': (np.mean(train_loss), np.std(train_loss)),
            'validation': (np.mean(valid_loss), np.std(valid_loss)),
            'test': (np.mean(test_loss), np.std(test_loss)),
        }

        ax_client.complete_trial(trial_index=trial_index,
                                 raw_data=raw_data)
        hps['trial_time'] = time.time() - startT
        print(f"{i+1} trial: {hps['trial_time']}")
        #torch.save(model.state_dict(), f"DenoiseNet-{i}.pt")
        run_a_sample_ax(model, validationSet, testSet, device, ax_trial = f"output-{i}")

    res = ax_client.get_trials_data_frame().sort_values('validation', ascending=False)
    res.to_csv('ax-stats-'+FLAGS.o_name+'.csv')

    #contour_plot = ax_client.get_contour_plot(param_x="lr", param_y="weight_decay", metric_name="validation")
    #print(type(contour_plot))

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    app.run(main)
