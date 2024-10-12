import os

import zarr
import numpy as np
import torch
from torchvision import transforms

from models import GMMGradient, RecoveryPolicy
from sklearn.mixture import GaussianMixture
from utils import draw_ood_latent, eval_encoder, unnormalize_data, normalize_data, get_data_stats

import matplotlib.pyplot as plt


def test_ood(cfg):
    data = zarr.open(cfg['ood_datapath'],'r')
    kp_dataset = np.array(data['data']['keypoint']).reshape(-1,18)
    end_indices = np.array(data['meta']['episode_ends']).astype(int)

    gmms = []
    for i in range(cfg["action_dim"]//cfg["space_dim"]):
        gmms.append(GaussianMixture(n_components=cfg["n_components"]))

    gmms_params = np.load(os.path.join(cfg['testing_dir'], "gmms.npz"), allow_pickle=True)
    for i in range(len(gmms_params)):
        for (key,val) in gmms_params[str(i)][()].items():
            setattr(gmms[i], key, val)

    rec_policy = GMMGradient(gmms_params)

    gmms_latent = []
    MVNs = []
    for i in range(len(gmms)):
        gmm = gmms[i]
        gmm_latent, _ = gmm.sample(300)
        gmms_latent.append(gmm_latent)
        MVNs.append([gmm.means_, gmm.covariances_])
    gmms_latent = np.hstack((gmms_latent))

    test_every = 5
    ood_latent = kp_dataset[::test_every]

    grad_arrows = []
    for i in range(ood_latent.shape[0]):
        dens, grad = rec_policy(ood_latent[i].reshape(1,-1,2))
        scaled_grad = (1-dens)*grad
        grad_arrows.append(scaled_grad[0])

    grad_arrows = np.vstack((grad_arrows))
    grad_arrows = np.hstack((ood_latent, grad_arrows))

    draw_ood_latent(gmms_latent, 
                    ood_latent, 
                    os.path.join(cfg['testing_dir'], "ood_latent.png"), 
                    grad_arrows=grad_arrows, 
                    MVNs=MVNs)
    


def latent_gradient_ascent():
    pass