import os
from datetime import datetime

import zarr
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


from dataset import ObsActPairs
from models import EquivalenceMap
from utils import get_data_stats, normalize_data
from src.model.full import GmmFull
from sklearn.mixture import GaussianMixture
from utils import draw_latent, eval_encoder



def train_recovery_policy(cfg):
    # config
    now = datetime.now()
    outpath = os.path.join(cfg["output_dir"], cfg["dataname"]+"; "+now.strftime("%m-%d-%Y_%H:%M"))
    os.makedirs(outpath, exist_ok=False)

    # data
    encoder_dataset = ObsActPairs(datapath=cfg["datapath"])
    encoder_loader = DataLoader(encoder_dataset,
                                batch_size=cfg["batch_size"],
                                num_workers=cfg["num_workers"],
                                shuffle=True,
                                pin_memory=True)
    data = zarr.open(cfg["datapath"],'r')
    image_dataset = np.array(data['data']['img'])[:,:,:,[2,1,0]]
    end_indices = np.array(data['meta']['episode_ends']).astype(int)
    test_image_dataset = image_dataset[:end_indices[cfg['num_test_traj']-1]]

    # model
    encoder = EquivalenceMap(input_size=cfg["input_size"], output_size=cfg["action_dim"])
    # gmm = GmmFull(num_components=cfg["n_components"], num_dims=cfg["action_dim"])
    gmm = GaussianMixture(n_components=cfg["n_components"])

    if torch.cuda.is_available():
        encoder.cuda()
        # gmm.cuda()
        device = 'cuda'
    else:
        device = 'cpu'

    encoder_optim = torch.optim.Adam(
        encoder.parameters(),
        lr=cfg['encoder_lr'])

    if cfg['load_encoder']:
        print('Loading Encoder from Checkpoint')
        encoder_ckpt = torch.load(os.path.join(cfg['testing_dir'], "encoder.pt"))
        encoder.load_state_dict(encoder_ckpt['model_state_dict'])
    else:
        # training encoder
        with tqdm(range(cfg['encoder_max_epoch']), desc='Epoch', leave=True) as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                encoder.train()
                epoch_loss = []
                # batch loop
                with tqdm(encoder_loader, desc='Batch', leave=False) as tepoch:
                    for batch in tepoch:
                        encoder_optim.zero_grad()

                        s, s_prime, action = batch
                        s, s_prime, action = s.to(device), s_prime.to(device), action.to(device)

                        z, z_prime = encoder(s), encoder(s_prime)

                        loss = encoder.action_loss(z, z_prime, action)
                        loss.backward()
                        encoder_optim.step()

                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)

                tglobal.set_postfix(loss=np.mean(epoch_loss))
                if epoch_idx%cfg['test_every'] == 0:
                    test_latent = eval_encoder(test_image_dataset, encoder, device)
                    draw_latent(test_latent, os.path.join(outpath, "test_latent.png"))
                if np.mean(epoch_loss) < cfg['encoder_loss_stop_threshold']:
                    print('Reached Loss Stop Threshold. Stopping Early.')
                    break
        # save encoder
        torch.save({
            'model_state_dict'              : encoder.state_dict(),
            }, os.path.join(outpath, "encoder.pt"))

    # latent data for gmm
    full_latent_data = eval_encoder(image_dataset, encoder, device).cpu().numpy()
    latent_stats = get_data_stats(full_latent_data)
    full_latent_data = normalize_data(full_latent_data, latent_stats)

    stats = {"latent_stats": latent_stats}
    np.savez(os.path.join(outpath, "stats.npz"), **stats)


    # fit gmm
    print("Fitting GMM.")
    # gmm.fit(full_latent_data,
    #         num_iterations=20000,
    #         mixture_lr=1e1,
    #         component_lr=1e1,
    #         log_freq=100)

    gmm.fit(full_latent_data)
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
    # visualize gmm
    print("Visualizing GMM.")
    gmm_latent, _ = gmm.sample(500)
    # draw_latent(full_latent_data.cpu()[::2], os.path.join(outpath, "full_latent.png"))
    draw_latent(gmm_latent, os.path.join(outpath, "gmm_latent.png"))
    
    # save gmm
    np.savez(os.path.join(outpath, "gmm.npz"), **gmm_params)
    # torch.save({
    #     'model_state_dict'              : gmm.state_dict(),
    #     }, os.path.join(outpath, "gmm.pt"))


    


    
