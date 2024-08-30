import os
from datetime import datetime

import h5py
import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from utils import draw_latent, robosuite_data_to_obj_dataset, to_obj_pose, gen_keypoints, get_data_stats



def train_recovery_policy(cfg):
    # config
    now = datetime.now()
    outpath = os.path.join(cfg["output_dir"], cfg["task"], cfg["datatype"], now.strftime("%m-%d-%Y_%H-%M-%S"))
    os.makedirs(outpath, exist_ok=False)

    # data
    data = h5py.File(cfg["datapath"], "r")
    object_dataset = robosuite_data_to_obj_dataset(data) #N,14
    N, _ = object_dataset.shape
    object_pose_dataset = to_obj_pose(object_dataset) #N,4,4
    object_kp_dataset = gen_keypoints(object_pose_dataset) #N,3,3
    object_kp_dataset = object_kp_dataset.reshape(N,-1)

    # model
    gmms = []
    for i in range(3):
        gmms.append(GaussianMixture(n_components=cfg["n_components"]))

    # fit gmm
    print("Fitting GMM.")
    gmms_params = {}
    for i in range(len(gmms)):
        gmm = gmms[i]
        gmm.fit(object_kp_dataset[:,3*i:3*(i+1)])
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
    gmms_latent = []
    for i in range(len(gmms)):
        gmm = gmms[i]
        gmm_latent, _ = gmm.sample(400)
        gmms_latent.append(gmm_latent)
    gmms_latent = np.hstack((gmms_latent))

    draw_latent(object_kp_dataset[::25], os.path.join(outpath, "full_latent.png"))
    draw_latent(gmms_latent, os.path.join(outpath, "gmms_latent.png"))
    
    # save gmm
    np.savez(os.path.join(outpath, "gmms.npz"), **gmms_params)



    


    
