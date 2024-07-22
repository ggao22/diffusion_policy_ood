import os

import zarr
import numpy as np
import torch
from torchvision import transforms

from models import EquivalenceMap, RecoveryPolicy
from sklearn.mixture import GaussianMixture
from utils import draw_ood_latent, eval_encoder, unnormalize_data

import matplotlib.pyplot as plt


def test_ood(cfg):
    data = zarr.open(cfg['ood_datapath'],'r')
    image_dataset = np.array(data['data']['img'])[:,:,:,[2,1,0]]
    # plt.imshow(image_dataset[0])
    # plt.show()
    end_indices = np.array(data['meta']['episode_ends']).astype(int)
    test_image_dataset = image_dataset[:end_indices[cfg['num_test_traj']-1]]
    preprocess = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    encoder = EquivalenceMap(input_size=cfg["input_size"], output_size=cfg["action_dim"])
    gmm = GaussianMixture(n_components=cfg["n_components"])

    if torch.cuda.is_available():
        encoder.cuda()
        device = 'cuda'
    else:
        device = 'cpu'


    encoder_ckpt = torch.load(os.path.join(cfg['testing_dir'], "encoder.pt"))
    encoder.load_state_dict(encoder_ckpt['model_state_dict'])

    gmm_params = np.load(os.path.join(cfg['testing_dir'], "gmm.npz"))
    for (key,val) in gmm_params.items():
        setattr(gmm, key, val)

    stats = np.load(os.path.join(cfg['testing_dir'], "stats.npz"), allow_pickle=True)
    latent_stats = stats['latent_stats'][()]
    rec_policy = RecoveryPolicy(encoder, gmm_params, latent_stats, eps=cfg['eps'], tau=cfg['tau'], eta=cfg['eta'])

    gmm_latent, _ = gmm.sample(250)
    gmm_latent = unnormalize_data(gmm_latent, latent_stats)
    test_every = 10
    ood_latent = eval_encoder(test_image_dataset[::test_every], encoder, device)
    grad_arrows = []
    for i in range(0,test_image_dataset.shape[0],test_every):
        _, grad = rec_policy(preprocess(test_image_dataset[i]).unsqueeze(0).to(device))
        grad_arrows.append(grad)
    grad_arrows = np.vstack((grad_arrows))
    grad_arrows = np.hstack((ood_latent, grad_arrows))

    draw_ood_latent(gmm_latent, 
                    ood_latent, 
                    os.path.join(cfg['testing_dir'], "ood_latent.png"), 
                    grad_arrows=grad_arrows, 
                    mu=unnormalize_data(gmm_params["means_"], latent_stats))


    