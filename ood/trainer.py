import os
from datetime import datetime

import zarr
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from dataset import ObsActPairs
from models import EquivalenceMap, GaussianMixture


def draw_3d_traj(zs, save_path):
    fig = plt.figure(figsize=(18,12))
    fig.subplots_adjust(top=0.92)
    fig.tight_layout()
    azims = [90]*6
    elevs = [50]*6
    for o in range(1,7):
        ax = fig.add_subplot(2, 3, o, projection='3d')
        x = zs[:,o-1]
        y = zs[:,o]
        z = zs[:,o+1]
        for i in range(len(x)):
            ax.scatter(x[i:i+1], y[i:i+1], z[i:i+1], color=plt.cm.rainbow(i/zs.shape[0]), s=30)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.azim = azims[o-1]
            ax.elev = elevs[o-1]
    plt.savefig(os.path.join(save_path, "latent.png"))

def draw_latent(zs, save_path):
    fig = plt.figure(figsize=(18,12))
    fig.tight_layout()
    for o in range(1,7):
        ax = fig.add_subplot(2, 3, o)
        x = zs[:,2*(o-1)]
        y = zs[:,2*(o-1)+1]
        for i in range(len(x)):
            ax.scatter(x[i:i+1], y[i:i+1], color=plt.cm.rainbow(i/zs.shape[0]), s=30)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f"Point {o}")
    plt.savefig(save_path)


def eval_encoder(imgs, encoder, device):
    with torch.no_grad():
        preprocess = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        latent_data = []
        for i in range(len(imgs)):
            latent_data.append(encoder(preprocess(imgs[i]).unsqueeze(0).to(device)).detach().cpu().numpy())
        latent_data = torch.from_numpy(np.vstack((latent_data)))
        return latent_data


def train_recovery_policy(cfg):
    # config
    now = datetime.now()
    outpath = os.path.join(cfg["output_dir"], now.strftime("%m-%d-%Y_%H:%M"))
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
    gmm = GaussianMixture(n_components=cfg["n_components"], n_features=cfg["action_dim"])

    if torch.cuda.is_available():
        encoder.cuda()
        gmm.cuda()
        device = 'cuda'
    else:
        device = 'cpu'

    encoder_optim = torch.optim.Adam(
        encoder.parameters(),
        lr=cfg['encoder_lr'])


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
    torch.save(encoder, os.path.join(outpath, "encoder.ckpt"))

    # latent data for gmm
    full_latent_data = eval_encoder(image_dataset[::2], encoder, device).to(device)

    # fit gmm
    print("Fitting GMM.")
    gmm.fit(full_latent_data)

    # visualize gmm
    print("Visualizing GMM.")
    gmm_latent, _ = gmm.sample(500)
    draw_latent(full_latent_data.cpu()[::2], os.path.join(outpath, "full_latent.png"))
    draw_latent(gmm_latent.cpu(), os.path.join(outpath, "gmm_latent.png"))
    
    # save gmm
    torch.save(gmm, os.path.join(outpath, "gmm.ckpt"))


    


    
