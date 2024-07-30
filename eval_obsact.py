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

from ood.utils import get_center_pos, get_center_ang, centralize, decentralize

import pygame
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
import numpy as np



def generate_kp_traj(kp,alpha,delay=12):
    kp_start = kp[0]
    kp_len = len(kp)
    kp_base = np.repeat([kp_start], kp_len, axis=0)

    motion_vec = (np.random.rand(2)-0.5)*alpha
    motion_vecs = np.repeat([motion_vec], kp_len-delay, axis=0)
    motion_vecs = np.vstack((np.zeros((delay, 2)),motion_vecs))
    vecs = np.repeat(np.cumsum(motion_vecs, axis=0), 9, axis=0).reshape(kp_len, 9, 2)
    return kp_base + vecs



@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-s', '--screen_size', default=512)
def main(checkpoint, output_dir, device, screen_size):
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

    testing = False
    if testing:
        cfg.task.dataset.zarr_path = 'data/pusht_demo_left_test.zarr'
    dataset: BaseLowdimDataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    tested_demos = 2
    ends = [0] + list(dataset.replay_buffer.episode_ends)
    kps = dataset.replay_buffer['keypoint'][:ends[tested_demos]]
    states = dataset.replay_buffer['state'][:ends[tested_demos]]
    acts = dataset.replay_buffer['action'][:ends[tested_demos]]
    
    pltscreen = (np.ones((screen_size,screen_size,3)) * 255).astype(int)
    fig, (ax1, ax2) = plt.subplots(1,2)

    def animate(args):
        kp_org, act_org, dcount, env_img = args
        kp = np.copy(kp_org)
        act = np.copy(act_org)
        ax1.cla()
        ax2.cla()
        ax1.imshow(pltscreen)
        ax2.imshow(env_img)
        ax1.set_title(f'Demo{dcount}')
        kp = kp.T
        width = (kp[:,5]+kp[:,6])//2 - (kp[:,1]+kp[:,2])//2
        ax1.fill(np.hstack((kp[0,[1,2]],np.flip(kp[0,[1,2]]+width[0]))), np.hstack((kp[1,[1,2]],np.flip(kp[1,[1,2]]+width[1]))), color='grey')
        kp[:,7] = kp[:,7] - width*0.90
        ax1.fill(np.hstack((kp[0,[4,7]],np.flip(kp[0,[4,7]]+width[1]))), np.hstack((kp[1,[4,7]],np.flip(kp[1,[4,7]]-width[0],))), color='grey')
        ee = plt.Circle(act, 10, color='b')
        ax1.add_patch(ee)
    
    # obs_in = []
    # act_out = []
    # env_imgs = []
    # demo_counts = []
    # for i in range(tested_demos):
    #     chunks = list(range(ends[i], ends[i+1], 16))
    #     for k in range(len(chunks)-1):
    #         demo_counts.append([i]*16)
    #         kp_chunk = kps[chunks[k]:chunks[k+1]] # 16,9,2
    #         center_pos = get_center_pos(kp_chunk[0])
    #         center_ang = get_center_ang(kp_chunk[0])
    #         centered_kp = centralize(kp_chunk, center_pos, center_ang, screen_size)
    #         obs_in.append(centered_kp)
            
    #         centered_act = centralize(acts[chunks[k]:chunks[k+1]], center_pos, center_ang, screen_size)
    #         act_out.append(centered_act)

    #         [env_imgs.append(pltscreen) for _ in range(16)]
    
    # ani = FuncAnimation(fig, animate, frames=zip(kps,acts,demo_counts), interval=100, save_count=sys.maxsize)
    # ani.save(os.path.join(output_dir,'obs_to_act_true.mp4'), writer='ffmpeg', fps=10) 

    # create PushT env with keypoints
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsEnv(render_size=512, render_action=False, **kp_kwargs)
    clock = pygame.time.Clock()


    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # policy
    obs_in = []
    act_out = []
    demo_counts = []
    env_imgs = []
    gen_random_traj = False
    for i in range(tested_demos):
        chunks = list(range(ends[i], ends[i+1], 16))
        
        for k in range(len(chunks)-1):
            demo_counts.append([i]*16)

            kp_chunk = kps[chunks[k]:chunks[k+1]] # 16,9,2
            center_pos = get_center_pos(kp_chunk[0])
            center_ang = get_center_ang(kp_chunk[0])
            kp_traj = centralize(kp_chunk, center_pos, center_ang, screen_size)

            act_chunk = acts[chunks[k]:chunks[k+1]] # 16,2
            init_action = centralize(act_chunk, center_pos, center_ang, screen_size)[0]

            if gen_random_traj:
                kp_traj = generate_kp_traj(kp_traj,8)
                init_action = [np.random.randint(50, 450), np.random.randint(50, 450),]
            np_obs_dict = {
                'obs': np.expand_dims(kp_traj.reshape(16,18),0).astype(np.float32),
                'init_action': np.expand_dims(init_action,0).astype(np.float32),
            }
            # obs_in.append(kp_traj)
            obs_in.append(decentralize(kp_traj, center_pos, center_ang, screen_size))


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
            
            np_action = decentralize(np_action_dict['action_pred'].squeeze(0), center_pos, center_ang, screen_size)
            act_out.append(np_action)


            # reset env and get observations (including info and render for recording)
            reset_state = states[chunks[k]]
            # reset_state[:4] = centralize(reset_state[:4].reshape(2,2), center_pos, center_ang, screen_size).reshape(4,)
            # reset_state[:2] = init_action
            # reset_state[4] = 0

            env.reset_to_state = reset_state
            obs = env.reset()
            info = env._get_info()
            img = env.render(mode='human')

            for act in np_action:
                # step env and render
                obs, reward, done, info = env.step(act)
                img = env.render(mode='human')
                env_imgs.append(img)
            
            # regulate control frequency
            control_hz = 10
            clock.tick(control_hz)

        
    obs_in = np.vstack((obs_in))
    act_out = np.vstack((act_out))
    demo_counts = np.array(demo_counts).flatten()
    ani = FuncAnimation(fig, animate, frames=zip(obs_in,act_out,demo_counts,env_imgs), interval=100, save_count=sys.maxsize)
    ani.save(os.path.join(output_dir,'obs_to_act_id.mp4'), writer='ffmpeg', fps=10) 
    plt.show()



if __name__ == '__main__':
    main()
