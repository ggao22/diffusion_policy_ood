import os

import zarr
import numpy as np
import torch
from torchvision import transforms

from models import EquivariantMap, RecoveryPolicy
from sklearn.mixture import GaussianMixture
from utils import draw_ood_latent, eval_encoder, unnormalize_data, normalize_data, get_data_stats

import matplotlib.pyplot as plt


def test_ood(cfg):
    data = zarr.open(cfg['ood_datapath'],'r')
    image_dataset = np.array(data['data']['img'])[:,:,:,[2,1,0]]
    kp_dataset = np.array(zarr.open(cfg['datapath'],'r')['data']['keypoint']).reshape(-1,18)
    end_indices = np.array(data['meta']['episode_ends']).astype(int)
    test_image_dataset = image_dataset[:end_indices[cfg['num_test_traj']-1]]
    preprocess = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    # preprocess = transforms.Compose([transforms.ToPILImage(),
    #                                     transforms.ToTensor(),
    #                                     # transforms.RandomRotation(degrees=(0,180)),
    #                                     transforms.RandomResizedCrop(size=(96, 96), scale=(0.6,0.6), ratio=(1,1), antialias=True),
    #                                     ])
    encoder = EquivariantMap(input_size=cfg["input_size"], output_size=cfg["action_dim"])
    gmms = []
    for i in range(cfg["action_dim"]//cfg["space_dim"]):
        gmms.append(GaussianMixture(n_components=cfg["n_components"]))

    if torch.cuda.is_available():
        encoder.cuda()
        device = 'cuda'
    else:
        device = 'cpu'


    encoder_ckpt = torch.load(os.path.join(cfg['testing_dir'], "encoder.pt"))
    encoder.load_state_dict(encoder_ckpt['model_state_dict'])

    gmms_params = np.load(os.path.join(cfg['testing_dir'], "gmms.npz"), allow_pickle=True)
    for i in range(len(gmms_params)):
        for (key,val) in gmms_params[str(i)][()].items():
            setattr(gmms[i], key, val)

    stats = np.load(os.path.join(cfg['testing_dir'], "stats.npz"), allow_pickle=True)
    latent_stats = stats['latent_stats'][()]

    kp_stats = get_data_stats(kp_dataset)
    # action_space_stats = {'min': np.zeros_like(latent_stats['min']),
    #                       'max': np.ones_like(latent_stats['max']) * 512}

    rec_policy = RecoveryPolicy(encoder, gmms_params, latent_stats, eps=cfg['eps'], tau=cfg['tau'], eta=cfg['eta'])

    gmms_latent = []
    MVNs = []
    for i in range(len(gmms)):
        gmm = gmms[i]
        gmm_latent, _ = gmm.sample(500)
        gmms_latent.append(gmm_latent)
        sub_latent_stats = {}
        for key in kp_stats.keys():
            sub_latent_stats[key] = kp_stats[key][2*i:2*(i+1)]
        MVNs.append([unnormalize_data(gmm.means_, sub_latent_stats), gmm.covariances_])
    gmms_latent = np.hstack((gmms_latent))

    gmms_latent = unnormalize_data(gmms_latent, kp_stats)

    test_every = 2
    ood_latent = eval_encoder(test_image_dataset[::test_every], encoder, device)
    ood_latent = normalize_data(ood_latent, latent_stats)
    ood_latent = unnormalize_data(ood_latent, kp_stats)

    grad_arrows = []
    for i in range(0,test_image_dataset.shape[0],test_every):
        dens, grad = rec_policy(preprocess(test_image_dataset[i]).unsqueeze(0).to(device))
        grad_arrows.append((1-dens)*grad)

    grad_arrows = np.vstack((grad_arrows))
    grad_arrows = np.hstack((ood_latent, grad_arrows))

    draw_ood_latent(gmms_latent, 
                    ood_latent, 
                    os.path.join(cfg['testing_dir'], "ood_latent.png"), 
                    grad_arrows=grad_arrows, 
                    MVNs=MVNs)
    


def latent_gradient_ascent():
    pass