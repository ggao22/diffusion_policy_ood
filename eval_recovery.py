"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from torch.utils.data import DataLoader
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv

import pygame
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
import numpy as np
from ood.config import cfg as rec_cfg


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    pltscreen = (np.ones((512,512,3)) * 255).astype(int)
    fig, (ax1, ax2) = plt.subplots(1,2)

    def animate(args):
        kp_org, act_org, env_img = args
        kp = np.copy(kp_org)
        act = np.copy(act_org)
        ax1.cla()
        ax2.cla()
        ax1.imshow(pltscreen)
        ax2.imshow(env_img)
        kp = kp.T
        width = (kp[:,5]+kp[:,6])//2 - (kp[:,1]+kp[:,2])//2
        ax1.fill(np.hstack((kp[0,[1,2]],np.flip(kp[0,[1,2]]+width[0]))), np.hstack((kp[1,[1,2]],np.flip(kp[1,[1,2]]+width[1]))), color='grey')
        kp[:,7] = kp[:,7] - width*0.90
        ax1.fill(np.hstack((kp[0,[4,7]],np.flip(kp[0,[4,7]]+width[1]))), np.hstack((kp[1,[4,7]],np.flip(kp[1,[4,7]]-width[0],))), color='grey')
        ee = plt.Circle(act, 10, color='b')
        ax1.add_patch(ee)
    

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # create PushT env with keypoints
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsEnv(render_size=512, render_action=False,  display_rec=True, rec_cfg=rec_cfg, **kp_kwargs)
    clock = pygame.time.Clock()
    env.seed(73)
    env.rec_policy.eta = 1.0
    obs = env.reset()
    info = env._get_info()
    img = env.render(mode='human')

    # env policy control
    max_iter = 20
    action_horizon = 12
    obs_in = []
    act_out = []
    env_imgs = []
    for _ in range(max_iter):
        kps = obs[:18].reshape(9,2)
        rec_vec = info['rec_vec']
        kp_traj, kp_center = generate_kp_traj(kps, rec_vec, horizon=16, delay=8)

        np_obs_dict = {
            'obs': np.expand_dims(kp_traj.reshape(16,18),0).astype(np.float32),
        }
        obs_in.append(kp_traj[:action_horizon])

        # device transfer
        obs_dict = dict_apply(np_obs_dict, 
            lambda x: torch.from_numpy(x).to(
                device=device))

        # # run policy
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)

        # # device_transfer
        np_action_dict = dict_apply(action_dict,
            lambda x: x.detach().to('cpu').numpy())

        np_action = np_action_dict['action_pred'].squeeze(0)
        act_out.append(np_action[:action_horizon])

        # step env and render
        for i in range(action_horizon):
            # step env and render
            act = np_action[i] + kp_center
            obs, reward, done, info = env.step(act)
            img = env.render(mode='human')
            env_imgs.append(img)

        # regulate control frequency
        control_hz = 10
        clock.tick(control_hz)

    obs_in = np.vstack((obs_in))
    act_out = np.vstack((act_out))
    ani = FuncAnimation(fig, animate, frames=zip(obs_in,act_out,env_imgs), interval=100, save_count=sys.maxsize)
    ani.save(os.path.join(output_dir,'recovery.mp4'), writer='ffmpeg', fps=10) 
    plt.show()


def generate_kp_traj(kp, recovery_vec, horizon, delay, alpha=0.5):
    kp_mean = kp.mean(axis=0)
    kp_center = kp_mean - np.array([512//2, 512//2])
    kp_start = kp - kp_center
    kp_base = np.repeat([kp_start], horizon, axis=0)

    mean_recovery_vec = recovery_vec.mean(axis=0) * alpha
    motion_vecs = np.repeat([mean_recovery_vec], horizon-delay, axis=0)
    motion_vecs = np.vstack((np.zeros((delay, 2)),motion_vecs))
    vecs = np.repeat(np.cumsum(motion_vecs, axis=0), 9, axis=0).reshape(horizon, 9, 2)
    return kp_base + vecs, kp_center


if __name__ == '__main__':
    main()
