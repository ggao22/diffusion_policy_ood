import os

import zarr
import numpy as np
import torch
import h5py

from models import GMMGradient
from sklearn.mixture import GaussianMixture
from utils import draw_ood_latent, robosuite_data_to_obj_dataset, to_obj_pose, gen_keypoints, get_data_stats

import matplotlib.pyplot as plt


def test_rand(cfg):
    data = h5py.File(cfg["datapath"], "r")
    object_dataset = robosuite_data_to_obj_dataset(data) #N,14
    N, _ = object_dataset.shape
    object_pose_dataset = to_obj_pose(object_dataset) #N,4,4
    object_kp_dataset = gen_keypoints(object_pose_dataset) #N,3,3
    object_kp_dataset = object_kp_dataset.reshape(N,-1)

    kp_stats = get_data_stats(object_kp_dataset)
    for key in kp_stats.keys():
        kp_stats[key] = kp_stats[key].reshape(-1,3)
    box_min = np.min(kp_stats['min'], axis=0)
    box_max = np.max(kp_stats['max'], axis=0)
    rand_pts = torch.from_numpy(np.random.uniform(box_min,box_max,(cfg['n_pts_tested']*3,3)).reshape(-1,3,3))

    gmms = []
    for i in range(3):
        gmms.append(GaussianMixture(n_components=cfg["n_components"]))

    gmms_params = np.load(os.path.join(cfg['testing_dir'], "gmms.npz"), allow_pickle=True)
    for i in range(len(gmms_params)):
        for (key,val) in gmms_params[str(i)][()].items():
            setattr(gmms[i], key, val)

    rec_policy = GMMGradient(gmms_params)

    gmms_latent = []
    for i in range(len(gmms)):
        gmm = gmms[i]
        gmm_latent, _ = gmm.sample(400)
        gmms_latent.append(gmm_latent)
    gmms_latent = np.hstack((gmms_latent))

    dens, grad = rec_policy(rand_pts)
    rand_pts = rand_pts.reshape(-1,np.prod(rand_pts.shape[1:]))
    grad_arrows = np.hstack((rand_pts, (1-dens)*grad))

    draw_ood_latent(gmms_latent, 
                    rand_pts, 
                    os.path.join(cfg['testing_dir'], "ood_latent.png"), 
                    grad_arrows=grad_arrows, 
                    )
    
    
    
