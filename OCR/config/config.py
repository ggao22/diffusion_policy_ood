#### stores config for OCR ####
# TODO: Rewrite these in hydra

cfg = {
        # policy training configs
        "task": "square_lowdim_abs_obsact", # replace with your task config name
        "datapath": '/home/george/diffusion_policy/data/robomimic/datasets/square/ph/low_dim_abs.hdf5', # replace with your dataset path

        # gmm configs
        "n_components": 6,
        'gmmpath': '/home/george/diffusion_policy/ood_3d/output/square/low_dim_abs/08-27-2024_17-14-47/gmms.npz',

        # recovery configs
        'alpha': 0.00005,
        'random_walk': 0.008, 

        # ckpts
        'recovery_ckpt': '/home/george/diffusion_policy/data/outputs/2024.09.11/08.56.33_train_diffusion_unet_lowdim_obsact_square_lowdim/checkpoints/epoch=0400-train_action_mse_error=0.003.ckpt',
        'base_ckpt': "/home/george/diffusion_policy/data/outputs/2024.09.05/16.12.10_train_diffusion_unet_lowdim_square_lowdim_abs/checkpoints/epoch=0250-test_mean_score=0.960.ckpt",
    }   



