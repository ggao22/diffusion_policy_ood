import os

import zarr
import numpy as np
import torch

from models import EquivalenceMap, GaussianMixture
from utils import draw_ood_latent, eval_encoder


def test_ood(cfg):
    data = zarr.open(cfg['ood_datapath'],'r')
    image_dataset = np.array(data['data']['img'])[:,:,:,[2,1,0]]
    end_indices = np.array(data['meta']['episode_ends']).astype(int)
    test_image_dataset = image_dataset[:end_indices[cfg['num_test_traj']-1]]

    encoder = EquivalenceMap(input_size=cfg["input_size"], output_size=cfg["action_dim"])
    gmm = GaussianMixture(n_components=cfg["n_components"], n_features=cfg["action_dim"])

    if torch.cuda.is_available():
        encoder.cuda()
        gmm.cuda()
        device = 'cuda'
    else:
        device = 'cpu'


    encoder_ckpt = torch.load(os.path.join(cfg['testing_dir'], "encoder.pt"))
    encoder.load_state_dict(encoder_ckpt['model_state_dict'])

    gmm_ckpt = torch.load(os.path.join(cfg['testing_dir'], "gmm.pt"))
    gmm.load_state_dict(gmm_ckpt['model_state_dict'])

    gmm_latent, _ = gmm.sample(500)
    ood_latent = eval_encoder(test_image_dataset, encoder, device)
    draw_ood_latent(gmm_latent, ood_latent, os.path.join(cfg['testing_dir'], "ood_latent.png"))


    