import torch.cuda
from absl import flags, app
from absl.flags import FLAGS
from ax.service.ax_client import AxClient
from ax.plot.contour import plot_contour
from ax.plot.render import plot_config_to_html
from ax.utils.report.render import render_report_elements
from ax.plot.base import AxPlotTypes
from axdata3d.utils import *
from axdata3d.gan_models import *
from axdata3d.train3d import train
from axdata3d.get3data import get3Data
from torchinfo import summary
import wandb
import time, logging


def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hps = hps_initialization()
    ax_client = AxClient()
    ax_client.create_experiment(
        name="optimize_hps",
        parameters=[
            {
                "name": "depth",
                "type": "range",
                "bounds": [2, 4],
                "log_scale": False
            },
            {
                "name": "learning_rateG",
                "type": "range",
                "bounds": [1e-4, 1e-2],
                "log_scale": True
            },
            {
                "name": "learning_rateD",
                "type": "range",
                "bounds": [1e-4, 1e-2],
                "log_scale": True
            },
            {
                "name": "nchan",
                "type": "range",
                "bounds": [32, 64],
                "log_scale": False
            },
            {
                "name": "ndf",
                "type": "range",
                "bounds": [64, 128],
                "log_scale": False
            },
            {
                "name": "weight_decayD",
                "type": "range",
                "bounds": [0.01, 0.4],
                "log_scale": True
            },
            {
                "name": "weight_decayG",
                "type": "range",
                "bounds": [0.01, 0.4],
                "log_scale": True
            },
            {
                "name": "pixel_lambda",
                "type": "range",
                "bounds": [1., 2000.],
                "log_scale": True
            },
            {
                "name": "perc_lambda_1",
                "type": "range",
                "bounds": [.001, 5.],
                "log_scale": True
            },
            {
                "name": "perc_lambda_2",
                "type": "range",
                "bounds": [.001, 5.],
                "log_scale": True
            },
            {
                "name": "perc_lambda_3",
                "type": "range",
                "bounds": [.001, 5.],
                "log_scale": True
            },
            {
                "name": "perc_lambda_4",
                "type": "range",
                "bounds": [.001, 5.],
                "log_scale": True
            },
            {
                "name": "perc_lambda_5",
                "type": "range",
                "bounds": [.001, 5.],
                "log_scale": True
            }
        ],
        objective_name='testing', minimize=True,
    )
    ax_time = []
    for i in range(50):
        if not exists(f'./trained_models/AxTrial-{i}'):
            os.mkdir(f'./trained_models/AxTrial-{i}')
        startT = time.time()
        parameters, trial_index = ax_client.get_next_trial()
        hps.update(parameters)
        wandb.init(project="tiftrc-3d", name=f'AxTrial-{i}', entity="chrivasileiou", config = hps)
        netG, netD = build_models(hps)
        summary(netG, input_data=torch.randn(1, 1, 64,64,64))
        summary(netD, input_data=torch.randn(1, 1, 64,64,64))
        results, netG, netD, training_typo = train(netG,
                                                   netD,
                                                   hps=hps,
                                                   ax_trial=i)
        run_a_3d_sample(netG2=None, features=None, labels=None, hps=hps, training_typo=training_typo, device=device, n_samples=10, ax_trial=i)
        raw_data = {
            'testing': (np.mean(results['RMSE']), np.std(results['RMSE'])),
        }
        ax_client.complete_trial(trial_index=trial_index,
                                 raw_data=raw_data)
        ax_time.append(time.time() - startT)
        hps['trial_time'] = ax_time[-1]
        logging.info(f"{i} trial: {get_time(ax_time[-1])}")
        wandb.finish()

    res = ax_client.get_trials_data_frame().sort_values('testing', ascending=True)
    res.to_csv('ax-stats-' + FLAGS.o_name + '.csv')
    logging.info(get_time(np.array(ax_time).sum()))
    model = ax_client.generation_strategy.model
    plot_config = plot_contour(model=model, metric_name='testing')
    # create an Ax report
    with open('ax-interact-contour-plot.html', 'w') as outfile:
        outfile.write(render_report_elements(
        "Report", 
        html_elements=[plot_config_to_html(plot_config)], 
        header=False,
        ))

    #fig.write_html('ax-interact-contour-plot.html')

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    app.run(main)
