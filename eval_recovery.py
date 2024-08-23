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
import zarr
from torch.utils.data import DataLoader
from torchvision import transforms

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.env.pusht.pusht_keypoints_image_env import PushTKeypointsImageEnv


from ood.utils import get_center_pos, get_center_ang, centralize, centralize_grad, decentralize, unnormalize_data, normalize_data, get_data_stats
from ood.models import EquivariantMap, Decoder

import pygame
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
import numpy as np
from ood.config import cfg as rec_cfg


def load_policy(ckpt, device, output_dir):
    # load checkpoint
    payload = torch.load(open(rec_cfg['obsact_ckpt'], 'rb'), pickle_module=dill)
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
    return policy


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

@click.command()
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-s', '--screen_size', default=512)
def main(output_dir, device, screen_size):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    policy = load_policy(rec_cfg['obsact_ckpt'], device, output_dir)
    encoder = EquivariantMap(input_size=rec_cfg["input_size"], output_size=rec_cfg["action_dim"])
    encoder_ckpt = torch.load(os.path.join(rec_cfg['testing_dir'], "encoder.pt"))
    encoder.load_state_dict(encoder_ckpt['model_state_dict'])
    # decoder = Decoder(in_dim=rec_cfg["action_dim"])
    # decoder_ckpt = torch.load(os.path.join(rec_cfg['testing_dir'], "decoder.pt"))
    # decoder.load_state_dict(decoder_ckpt['model_state_dict'])

    stats = np.load(os.path.join(rec_cfg['testing_dir'], "stats.npz"), allow_pickle=True)
    latent_stats = stats['latent_stats'][()]
    kp_dataset = np.array(zarr.open(rec_cfg['datapath'],'r')['data']['keypoint']).reshape(-1,18)
    kp_stats = get_data_stats(kp_dataset)
    # kp_stats = stats['kp_stats'][()]
    preprocess = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    pltscreen = (np.ones((screen_size,screen_size,3)) * 255).astype(int)
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
    


    # create PushT env with keypoints
    kp_kwargs = PushTKeypointsImageEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsImageEnv(render_size=screen_size, render_action=False,  display_rec=True, rec_cfg=rec_cfg, **kp_kwargs)
    clock = pygame.time.Clock()

    obs_in = []
    act_out = []
    env_imgs = []
    for n in range(6):
        seed = n+777
        # seed = n+380
        env.seed(seed)
        obs = env.reset()
        while not condition(obs['keypoints'], 512, 'right'):
            seed += 2**12
            env.seed(seed)
            obs = env.reset()
        info = env._get_info()
        img = env.render(mode='human')

        # env policy control
        max_iter = 20
        action_horizon = 12
        delay_cycle = 12
        delay_count = delay_cycle
        for _ in range(max_iter):
            
            latent = encoder(preprocess(np.moveaxis(obs['image'],0,2)[:,:,[2,1,0]]).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
            print()
            # print(latent)
            latent = normalize_data(latent, latent_stats)
            # print(latent)
            pseudo_kp = unnormalize_data(latent, kp_stats)
            print(pseudo_kp)
            # print(pseudo_kp.reshape(9,2).mean(0))
            print(obs['keypoints'][:18])
            # pseudo_kp = decoder(torch.from_numpy(pseudo_kp).unsqueeze(0)).squeeze(0).detach().cpu().numpy()

            # kp = obs['keypoints'][:18].reshape(9,2)
            kp = pseudo_kp.reshape(9,2)
            rec_vec = info['rec_vec']

            if np.linalg.norm(rec_vec.mean(axis=0)) < 20: 
                print('recovery done.')
                break

            center_pos = get_center_pos(kp)
            kp_start = centralize(kp, center_pos, screen_size=screen_size) #9,2

            if delay_count==2:
                delay_count = delay_cycle
            kp_traj = generate_kp_traj(kp_start, rec_vec, horizon=16, delay=delay_count, alpha=0.055)
            delay_count -= 2

            init_action = centralize(np.expand_dims(info['pos_agent'],0), center_pos, screen_size=screen_size)

            np_obs_dict = {
                'obs': np.expand_dims(kp_traj.reshape(16,18),0).astype(np.float32),
                'init_action': init_action.astype(np.float32)
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
            np_action = decentralize(np_action, center_pos, screen_size=screen_size)

            # step env and render
            for i in range(action_horizon):
                # step env and render
                act = np_action[i]
                obs, reward, done, info = env.step(act)
                img = env.render(mode='human')
                env_imgs.append(img)

            # regulate control frequency
            control_hz = 5
            clock.tick(control_hz)

    obs_in = np.vstack((obs_in))
    act_out = np.vstack((act_out))
    ani = FuncAnimation(fig, animate, frames=zip(obs_in,act_out,env_imgs), interval=50, save_count=sys.maxsize)
    ani.save(os.path.join(output_dir,'rec2.mp4'), writer='ffmpeg', fps=20) 
    plt.show()


def generate_kp_traj(kp_start, recovery_vec, horizon, delay, alpha=0.01):
    kp_base = np.repeat([kp_start], horizon, axis=0) # horizon,9,2
    mean_recovery_vec = recovery_vec.mean(axis=0) * alpha
    motion_vecs = np.repeat([mean_recovery_vec], horizon-delay, axis=0) 
    motion_vecs = np.vstack((np.zeros((delay, 2)),motion_vecs)) # horizon,2
    vecs = np.repeat(np.cumsum(motion_vecs, axis=0), 9, axis=0).reshape(horizon, 9, 2)
    return kp_base + vecs


if __name__ == '__main__':
    main()
