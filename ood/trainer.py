import os
from datetime import datetime

import zarr
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


from dataset import ObsActPairs, ObsPosPairs, ObsActPosPairs
from models import EquivariantMap
from utils import get_data_stats, normalize_data
from sklearn.mixture import GaussianMixture
from utils import draw_latent, eval_encoder



def train_recovery_policy(cfg):
    # config
    now = datetime.now()
    outpath = os.path.join(cfg["output_dir"], cfg["dataname"], now.strftime("%m-%d-%Y_%H-%M-%S"))
    os.makedirs(outpath, exist_ok=False)

    data = zarr.open(cfg["datapath"],'r')
    keypoints = np.array(data['data']['keypoint'])
    keypoints = keypoints.reshape(len(keypoints), -1)

    gmms = []
    for i in range(cfg["action_dim"]//cfg["space_dim"]):
        gmms.append(GaussianMixture(n_components=cfg["n_components"]))


    # fit gmm
    print("Fitting GMM.")
    gmms_params = {}
    for i in range(len(gmms)):
        gmm = gmms[i]
        gmm.fit(keypoints[:,cfg["space_dim"]*i:cfg["space_dim"]*(i+1)])
        gmm_params = {
            "weights_": gmm.weights_,
            "means_": gmm.means_,
            "covariances_": gmm.covariances_,
            "precisions_": gmm.precisions_,
            "precisions_cholesky_": gmm.precisions_cholesky_,
            "converged_": gmm.converged_,
            "n_iter_": gmm.n_iter_,
            "lower_bound_": gmm.lower_bound_,
            "n_features_in_": gmm.n_features_in_,
        }
        gmms_params[str(i)] = gmm_params
    
    # visualize gmm
    print("Visualizing GMM.")
    gmms_vis = []
    for i in range(len(gmms)):
        gmm = gmms[i]
        gmm_latent, _ = gmm.sample(300)
        gmms_vis.append(gmm_latent)
    gmms_vis = np.hstack((gmms_vis))

    draw_latent(keypoints[::4], os.path.join(outpath, "full_latent.png"))
    draw_latent(gmms_vis, os.path.join(outpath, "gmms_vis.png"))
    
    # save gmm
    np.savez(os.path.join(outpath, "gmms.npz"), **gmms_params)



    


    
