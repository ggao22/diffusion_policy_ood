"""
Usage:
python eval.py -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
sys.path.append('../')

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.pusht.pusht_keypoints_image_env import PushTKeypointsImageEnv

from ood.utils import get_center_pos, get_center_ang, centralize, centralize_grad, decentralize

import pygame
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
import numpy as np
from ood.config import cfg as rec_cfg

# continuous learning
from diffusion_policy.common.replay_buffer import ReplayBuffer


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
@click.option('-a', '--augment_file', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(output_dir, augment_file, device):
    screen_size = 512

    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load translator policy from checkpoint
    translator_policy, translator_cfg = load_policy(rec_cfg['obsact_ckpt'], device, output_dir)
    # load base policy from checkpoint
    base_policy, base_cfg = load_policy(rec_cfg['base_ckpt'], device, output_dir)

    replay_buffer = ReplayBuffer.create_from_path(augment_file, mode='a')

    # create PushT env with keypoints
    kp_kwargs = PushTKeypointsImageEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsImageEnv(render_size=96, render_action=False,  display_rec=False, rec_cfg=rec_cfg, **kp_kwargs)
    clock = pygame.time.Clock()

    n_obs_steps = base_cfg.n_obs_steps
    max_iter = 60
    episodes = 5
    ood_threshold = 90

    for n in range(episodes):
        # CL
        episode = list()

        seed = n+787787
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
            print(np.linalg.norm(info['rec_vec'].mean(axis=0)))
            if np.linalg.norm(info['rec_vec'].mean(axis=0)) > ood_threshold:
                # case: Out-Of-Distribution
                kp = obs['keypoints'][:18].reshape(9,2)
                rec_vec = info['rec_vec']

                center_pos = get_center_pos(kp)
                center_ang = get_center_ang(kp)
                kp_start = centralize(kp, center_pos, center_ang, screen_size) #9,2
                rec_vec = centralize_grad(rec_vec, center_ang) #9,2
                kp_traj = generate_kp_traj(kp_start, rec_vec, horizon=16, delay=11, alpha=4.0)

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

                # step env and render
                for i in range(len(action)):
                    act = action[i]

                    # state dim 2+3
                    state = np.concatenate([info['pos_agent'], info['block_pose']])
                    keypoint = obs['keypoints'][:18].reshape(9,2)
                    data = {
                        'img': img,
                        'state': np.float32(state),
                        'keypoint': np.float32(keypoint),
                        'action': np.float32(act),
                        'n_contacts': np.float32([info['n_contacts']])
                    }
                    episode.append(data)

                    # step env and render
                    obs, reward, done, info = env.step(act)
                    img = env.render(mode='human')

                # regulate control frequency
                control_hz = 10
                clock.tick(control_hz)

            else: 
                break
        
        if click.confirm('Save Episode?'):
            # save episode buffer to replay buffer (on disk)
            data_dict = dict()
            for key in episode[0].keys():
                data_dict[key] = np.stack(
                    [x[key] for x in episode])
            replay_buffer.add_episode(data_dict, compressors='disk')
            print(f'saved seed {seed}')


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
