from trainer import train_recovery_policy


if __name__ == "__main__":
    cfg = {
    "output_dir": "output",
    "input_size": (96,96),
    "datapath": '../data/pusht_demo_50.zarr',
    "action_dim": 18,

    "num_workers": 2,
    "batch_size": 64,

    "num_test_traj": 3,
    "n_components": 400,

    "encoder_max_epoch": 600,
    "encoder_loss_stop_threshold": 1.5e+3,
    "encoder_lr": 5e-4,
    "test_every": 50,
    }   
    
    train_recovery_policy(cfg)