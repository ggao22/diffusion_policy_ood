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


def draw_ood_latent(zs, ood_zs, save_path, grad_arrows=np.array(None), mu=np.array(None)):
    fig = plt.figure(figsize=(18,12))
    fig.tight_layout()
    for o in range(1,7):
        ax = fig.add_subplot(2, 3, o)
        x = zs[:,2*(o-1)]
        y = zs[:,2*(o-1)+1]
        oodx = ood_zs[:,2*(o-1)]
        oody = ood_zs[:,2*(o-1)+1]
        ax.scatter(x, y, color='tab:blue', s=30)
        ax.scatter(oodx, oody, color='tab:red', s=30)
        if grad_arrows.any():
            gx = grad_arrows[:,2*(o-1)]
            gy = grad_arrows[:,2*(o-1)+1]
            gu = grad_arrows[:,2*(o-1)+18]
            gv = grad_arrows[:,2*(o-1)+1+18]
            ax.quiver(gx, gy, gu, gv, angles='xy', scale_units='xy', scale=5, alpha=0.6)
        if mu.any():
            mux = mu[:,2*(o-1)]
            muy = mu[:,2*(o-1)+1]
            ax.scatter(mux, muy, color='tab:orange', s=100, alpha=0.6)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f"Point {o}")
    plt.savefig(save_path)



# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    return ndata

def unnormalize_data(ndata, stats):
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def unnormalize_gradient(ndata, stats):
    data = ndata * (stats['max'] - stats['min'])
    return data

