"""
Usage:
python eval.py -o data/pusht_eval_output
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
import zarr
from torch.utils.data import DataLoader
from torchvision import transforms

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.pusht.pusht_keypoints_image_env import PushTKeypointsImageEnv

from ood.utils import get_center_pos, get_center_ang, centralize, centralize_grad, decentralize, unnormalize_data, normalize_data, get_data_stats
from ood.models import EquivariantMap

import pygame
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
import numpy as np
from ood.config import cfg as rec_cfg


def load_policy(ckpt, device, output_dir):
    # load checkpoint
    payload = torch.load(open(ckpt, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    return policy, cfg


def condition(kps, side_len, pre_def='left'):
    if pre_def=='left':
        cond = lambda x: x > side_len//2
    elif pre_def=='right':
        cond = lambda x: x < side_len//2
    else:
        pass

    keypoint = kps.reshape(2,-1)[0].reshape(-1,2)[:9]
    for pt in keypoint:
        if cond(pt[0]):
            return False
    return True


def add_obs(new_obs, past_obs, n_obs_steps):
    for key in past_obs.keys():
        obs_item = np.expand_dims(new_obs[key],(0,1))
        if len(past_obs[key]) < 1:
            past_obs[key] = np.repeat(obs_item, n_obs_steps, 1)
        else:
            old_obs = past_obs[key][:, 1:]
            past_obs[key] = np.hstack((old_obs,obs_item))
    return past_obs



@click.command()
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-s', '--screen_size', default=512)
def main(output_dir, device, screen_size):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load translator policy from checkpoint
    translator_policy, translator_cfg = load_policy(rec_cfg['obsact_ckpt'], device, output_dir)
    # load base policy from checkpoint
    base_policy, base_cfg = load_policy(rec_cfg['base_ckpt'], device, output_dir)

    stats = np.load(os.path.join(rec_cfg['testing_dir'], "stats.npz"), allow_pickle=True)
    latent_stats = stats['latent_stats'][()]
    kp_dataset = np.array(zarr.open(rec_cfg['datapath'],'r')['data']['keypoint']).reshape(-1,18)
    kp_stats = get_data_stats(kp_dataset)
    encoder = EquivariantMap(input_size=rec_cfg["input_size"], output_size=rec_cfg["action_dim"])
    encoder_ckpt = torch.load(os.path.join(rec_cfg['testing_dir'], "encoder.pt"))
    encoder.load_state_dict(encoder_ckpt['model_state_dict'])
    preprocess = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


    pltscreen = (np.ones((screen_size,screen_size,3)) * 255).astype(int)
    fig, ax = plt.subplots()

    def animate(args):
        env_img, state = args
        if state==0:
            fig.suptitle('Base Policy')
        else:
            fig.suptitle('Recovery Policy')
        ax.cla()
        ax.imshow(env_img)
    

    # create PushT env with keypoints
    kp_kwargs = PushTKeypointsImageEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsImageEnv(render_size=screen_size, render_action=False,  display_rec=True, rec_cfg=rec_cfg, **kp_kwargs)
    clock = pygame.time.Clock()

    n_obs_steps = base_cfg.n_obs_steps

    states = []
    env_imgs = []
    max_iter = 100
    episodes = 1
    ood_threshold = 40

    for n in range(episodes):
        # seed = n+350
        seed = n+380
        env.seed(seed)
        obs = env.reset()
        while not condition(obs['keypoints'], 512, 'right'):
            seed += 2**12
            env.seed(seed)
            obs = env.reset()
        info = env._get_info()
        img = env.render(mode='human')
        past_obs = {'image': [], 'agent_pos': []}
        past_obs = add_obs(obs, past_obs, n_obs_steps)

        kp = obs['keypoints'][:18].reshape(9,2)
        center_pos = get_center_pos(kp)
        center_ang = get_center_ang(kp)
        kp_start = centralize(kp, center_pos, center_ang, screen_size) #9,2

        # env policy control
        for iter in range(max_iter):
            # print(np.linalg.norm(info['rec_vec'].mean(axis=0)))

            latent = encoder(preprocess(np.moveaxis(obs['image'],0,2)).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
            latent = normalize_data(latent, latent_stats)
            pseudo_kp = unnormalize_data(latent, kp_stats)

            center_pos = get_center_pos(pseudo_kp.reshape(9,2))
            true_center_pos = get_center_pos(obs['keypoints'][:18].reshape(9,2))
            print(center_pos)
            print(true_center_pos)

            center_ang = get_center_ang(pseudo_kp.reshape(9,2))
            true_center_ang = get_center_ang(obs['keypoints'][:18].reshape(9,2))
            print(center_ang)
            print(true_center_ang)

            if np.linalg.norm(info['rec_vec'].mean(axis=0)) < ood_threshold and iter>0:
                # case: In-Distribution
                past_obs = add_obs(obs, past_obs, n_obs_steps)

                # device transfer
                obs_dict = dict_apply(past_obs, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = base_policy.predict_action(obs_dict)
                
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)

                states.extend([0]*action.shape[0])
                
            else:
                # case: Out-Of-Distribution
                latent = encoder(preprocess(np.moveaxis(obs['image'],0,2)).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
                latent = normalize_data(latent, latent_stats)
                pseudo_kp = unnormalize_data(latent, kp_stats)

                # kp = obs['keypoints'][:18].reshape(9,2)
                kp = pseudo_kp.reshape(9,2)
                rec_vec = info['rec_vec']

                center_pos = get_center_pos(kp)
                center_ang = get_center_ang(kp)
                kp_start = np.array([[264.05470922, 305.95380222],
                            [195.91772399, 213.134809  ],
                            [317.78647403, 215.73112323],
                            [256.,         241.25413635],
                            [239.97253574, 338.18345584],
                            [295.09468435, 249.46903473],
                            [216.14934828, 247.52549507],
                            [238.59240633, 278.18318234],
                            [280.43211806, 214.56496122]])

                # kp_start = centralize(kp, center_pos, center_ang, screen_size) #9,2
                rec_vec = centralize_grad(rec_vec, center_ang) #9,2
                kp_traj = generate_kp_traj(kp_start, rec_vec, horizon=16, delay=10, alpha=2.0)

                init_action = centralize(np.expand_dims(info['pos_agent'],0), center_pos, center_ang, screen_size)

                np_obs_dict = {
                    'obs': np.expand_dims(kp_traj.reshape(16,18),0).astype(np.float32),
                    'init_action': init_action.astype(np.float32)
                }

                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = translator_policy.predict_action(obs_dict)

                # # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action_pred'].squeeze(0)
                action = decentralize(action, center_pos, center_ang, screen_size)

                states.extend([1]*action.shape[0])


            # step env and render
            for i in range(len(action)):
                # step env and render
                act = action[i]
                obs, reward, done, info = env.step(act)
                img = env.render(mode='human')
                env_imgs.append(img)

                if done: break
            if done: 
                print('done')
                states = states[:len(env_imgs)]
                break

            # regulate control frequency
            control_hz = 10
            clock.tick(control_hz)

    print(len(env_imgs))
    print(len(states))
    ani = FuncAnimation(fig, animate, frames=zip(env_imgs,states), interval=50, save_count=sys.maxsize)
    ani.save(os.path.join(output_dir,'base_ood.mp4'), writer='ffmpeg', fps=20) 
    plt.show()


def generate_kp_traj(kp_start, recovery_vec, horizon, delay, alpha=0.01):
    kp_base = np.repeat([kp_start], horizon, axis=0) # horizon,9,2
    mean_recovery_vec = recovery_vec.mean(axis=0)
    mean_recovery_vec = mean_recovery_vec/np.linalg.norm(mean_recovery_vec) * alpha
    motion_vecs = np.repeat([mean_recovery_vec], horizon-delay, axis=0) 
    motion_vecs = np.vstack((np.zeros((delay, 2)),motion_vecs)) # horizon,2
    vecs = np.repeat(np.cumsum(motion_vecs, axis=0), 9, axis=0).reshape(horizon, 9, 2)
    return kp_base + vecs


if __name__ == '__main__':
    main()
