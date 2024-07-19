import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


# evaluating
def eval_encoder(imgs, encoder, device):
    with torch.no_grad():
        preprocess = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        latent_data = []
        for i in range(len(imgs)):
            latent_data.append(encoder(preprocess(imgs[i]).unsqueeze(0).to(device)).detach().cpu().numpy())
        latent_data = torch.from_numpy(np.vstack((latent_data)))
        return latent_data


# plotting
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


def draw_ood_latent(zs, ood_zs, save_path):
    fig = plt.figure(figsize=(18,12))
    fig.tight_layout()
    for o in range(1,7):
        ax = fig.add_subplot(2, 3, o)
        x = zs[:,2*(o-1)]
        y = zs[:,2*(o-1)+1]
        oodx = ood_zs[:,2*(o-1)]
        oody = ood_zs[:,2*(o-1)+1]
        for i in range(len(x)):
            ax.scatter(x[i:i+1], y[i:i+1], color='tab:blue', s=30)
            ax.scatter(oodx[i:i+1], oody[i:i+1], color='tab:red', s=30)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f"Point {o}")
    plt.savefig(save_path)