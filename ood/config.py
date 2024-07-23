# stores config for trainer, tester, and visualization files
cfg = {
        # main cfgs
        "output_dir": "output",
        "input_size": (96,96),
        "dataname": "pusht_demo_left",
        "action_dim": 18,
        "space_dim": 2,

        "num_workers": 2,
        "batch_size": 64,

        "num_test_traj": 8,
        "n_components": 4,
        
        "load_encoder": True,
        "encoder_max_epoch": 600,
        "encoder_loss_stop_threshold": 1.5e+3,
        "encoder_lr": 2e-4,
        "test_every": 50,

        # testing cfgs
        'ood_datapath': '/home/georgegao/diffusion_policy_ood/data/pusht_demo_right.zarr',
        'testing_dir': '/home/georgegao/diffusion_policy_ood/ood/output/pusht_demo_left_07-23-2024_15$21',

        # rec cfg
        # "eps": -0.0005,
        # "tau": 0.00004,
        "eps": -40,
        "tau": 7.5,
        "eta": 1.0
    }   

cfg["datapath"] = f'/home/georgegao/diffusion_policy_ood/data/{cfg["dataname"]}.zarr'