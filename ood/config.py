# stores config for trainer, tester, and visualization files
cfg = {
        # main cfgs
        'loss_type': 'action',
        "output_dir": "output",
        "input_size": (96,96),
        "dataname": "pusht_demo_left_v2",
        "action_dim": 18,
        "space_dim": 2,

        "num_workers": 2,
        "batch_size": 64,

        "num_test_traj": 2,
        "n_components": 2,
        
        "load_encoder": True,
        "encoder_max_epoch": 1000,
        "encoder_loss_stop_threshold": 1.0e+3,
        "encoder_lr": 1e-3,
        "test_every": 100,

        # testing cfgs
        'ood_datapath': '/home/georgegao/diffusion_policy_ood/data/pusht_demo_right_test.zarr',
        'testing_dir': '/home/georgegao/diffusion_policy_ood/ood/output/pusht_demo_left_07-23-2024_15$21',
        # 'testing_dir': '/home/georgegao/diffusion_policy_ood/ood/output/pusht_demo_left_v2_07-31-2024_11$46',

        # rec cfg
        "eps": -40,
        "tau": 7.5,
        "eta": 1.0,
        'obsact_ckpt': '/home/georgegao/diffusion_policy_ood/data/outputs/2024.07.30/12.20.59_train_diffusion_unet_lowdim_obsact_pusht_lowdim_obsact/checkpoints/epoch=3800-train_action_mse_error=207.801.ckpt',
        # 'base_ckpt': '/home/georgegao/diffusion_policy_ood/data/outputs/2024.08.01/10.00.59_train_diffusion_unet_image_pusht_image/checkpoints/epoch=0300-train_mean_score=0.996.ckpt',
        'base_ckpt': '/home/georgegao/diffusion_policy_ood/data/outputs/2024.08.01/10.00.59_train_diffusion_unet_image_pusht_image/checkpoints/latest.ckpt',
    }   

cfg["datapath"] = f'/home/georgegao/diffusion_policy_ood/data/{cfg["dataname"]}.zarr'


