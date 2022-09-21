from absl import flags, app
from absl.flags import FLAGS
from ax.service.ax_client import AxClient
from randpointslib.getData import getData
from randpointslib.model_ax_allHyperparams import *
from randpointslib.train_ax_allHyperparams import *
from matplotlib import pyplot
from pandas.plotting import table
import numpy
import time, statistics

def main(argv):
    del argv
    hps = {
        'batch_size': FLAGS.batch_size,
        'out_channels_0': out_channels[0],
        'out_channels_1': out_channels[1],
        'out_channels_2': out_channels[2],
        'out_channels_3': out_channels[3],
        'lr': FLAGS.learning_rate,
        'weight_decay': FLAGS.weight_decay,
        'n_conv_layers': 4,
        'momentum': 0.3,
        'schedFactor': FLAGS.schedFactor,
        'schedPatience': FLAGS.schedPatience,
        'n_epochs': 150,
        'optimizer': 'Adam',
        'criterion': 'GaussianNLLLoss',
        'trial_time': 0
    }
    
    ax_client = AxClient()
    print(type(ax_client))
    convhps_LUT = []
    for i in range(11):
        convhps_LUT.append({'padding': i, 'kernel_size': 2*i+1})
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
             "bounds": [2, 10],
             "log_scale": True
            },
            {
             "name": "weight_decay",
             "type": "range",
             "bounds": [1e-2, 1e-0],
             "log_scale": True
            },
            {
             "name": "n_conv_layers",
             "type": "range",
             "bounds": [2, 4],
             "log_scale": True
            },
            {
             "name": "out_channels_0",
             "type": "range",
             "bounds": [8, 64],
             "log_scale": True
            },
            {
             "name": "convhps_0",
             "type": "range",
             "bounds": [0, 10],
             "log_scale": False
            },
            {
             "name": "out_channels_1",
             "type": "range",
             "bounds": [8, 32],
             "log_scale": True
            },
            {
             "name": "convhps_1",
             "type": "range",
             "bounds": [0, 10],
             "log_scale": False
            },
            {
             "name": "out_channels_2",
             "type": "range",
             "bounds": [8, 32],
             "log_scale": True
            },
            {
             "name": "convhps_2",
             "type": "range",
             "bounds": [0, 10],
             "log_scale": False
            },
            {
             "name": "out_channels_3",
             "type": "range",
             "bounds": [4, 32],
             "log_scale": True
            },
            {
             "name": "convhps_3",
             "type": "range",
             "bounds": [0, 10],
             "log_scale": False
            }
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
        model = DenoiseNet_ax(hps, convhps_LUT, printForward = True)
        print(model)
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
        print(f"{i} trial: {hps['trial_time']}")
        #torch.save(model.state_dict(), f"DenoiseNet-{i}.pt")
        run_a_sample_ax(model, validationSet, testSet, device, ax_trial = f"output-{i}")

    res = ax_client.get_trials_data_frame().sort_values('validation', ascending=False)
    res.to_csv('ax-stats-'+FLAGS.o_name+'.csv')

    #contour_plot = ax_client.get_contour_plot(param_x="lr", param_y="weight_decay", metric_name="validation")
    #print(type(contour_plot))

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    app.run(main)
