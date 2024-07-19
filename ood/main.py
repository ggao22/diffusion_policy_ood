import click

from trainer import train_recovery_policy
from tester import test_ood



@click.command()
@click.option('-m', '--mode', required=True, type=str)
def main(mode):
    cfg = {
        # main cfgs
        "output_dir": "output",
        "input_size": (96,96),
        "dataname": "pusht_demo_left",
        "action_dim": 18,

        "num_workers": 2,
        "batch_size": 64,

        "num_test_traj": 8,
        "n_components": 400,

        "encoder_max_epoch": 600,
        "encoder_loss_stop_threshold": 1.5e+3,
        "encoder_lr": 5e-4,
        "test_every": 50,

        # testing cfgs
        'ood_datapath': '../data/pusht_demo_right.zarr',
        'testing_dir': 'output/pusht_demo_left; 07-19-2024_16:14',
    }   

    cfg["datapath"] = f'../data/{cfg["dataname"]}.zarr'

    if mode=='train':
        train_recovery_policy(cfg)
    elif mode=='test':
        test_ood(cfg)
    else:
        raise Exception() 



if __name__ == "__main__":
    main()