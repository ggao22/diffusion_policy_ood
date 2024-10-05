"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import os
from datetime import datetime

import h5py
import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from utils import draw_latent, robosuite_data_to_obj_dataset, to_obj_pose, gen_keypoints



def train_gmm(recovery_cfg, task_cfg):
    # config
    now = datetime.now()
    outpath = os.path.join('data/outputs', now.strftime("%Y.%m.%d"), now.strftime("%H.%M.%S") + recovery_cfg['name'] + '_gmm')
    os.makedirs(outpath, exist_ok=False)

    kp_dim = task_cfg['keypoint_dim']
    if kp_dim == 3:
        ### Robomimic Task ###
        # get keypoints from pose
        data = h5py.File(task_cfg["dataset_path"], "r")
        object_dataset = robosuite_data_to_obj_dataset(data) #N,14
        N, _ = object_dataset.shape
        object_pose_dataset = to_obj_pose(object_dataset) #N,4,4
        object_kp_dataset = gen_keypoints(object_pose_dataset) #N,3,3
        object_kp_dataset = object_kp_dataset.reshape(N,-1) #N,9

    # model
    gmms = []
    for i in range(object_kp_dataset.shape[1]//kp_dim):
        gmms.append(GaussianMixture(n_components=recovery_cfg["n_components"]))

    # fit gmm
    print("Fitting GMM.")
    gmms_params = {}
    for i in range(len(gmms)):
        gmm = gmms[i]
        gmm.fit(object_kp_dataset[:,kp_dim*i:kp_dim*(i+1)])
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



    


    



if __name__ == "__main__":
    main()
