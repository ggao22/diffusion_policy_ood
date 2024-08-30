"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
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
from torch.utils.data import DataLoader
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.pytorch_util import dict_apply
# from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv

from utils import to_obj_pose, gen_keypoints, abs_traj

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
import numpy as np





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

    cfg.task.dataset['dataset_path'] = '../' + cfg.task.dataset['dataset_path']

    dataset: BaseLowdimDataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    # print(dataset.replay_buffer.keys())
    episode_starts = [0] + list(dataset.replay_buffer.episode_ends[:-1])
    # print(dataset.replay_buffer['obs'][:episode_starts[1]][:,:3])

    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True)
    dataiter = iter(dataloader)
    # _ = next(dataiter)
    vis_batch = next(dataiter)
    kps = vis_batch['obs'][::4]
    kps = kps.reshape(-1,9)
    print(kps.shape)

    # eps = 1
    # obj_se3 = dataset.replay_buffer['obs'][episode_starts[eps]:episode_starts[eps+1]]
    # obj_pose = to_obj_pose(obj_se3)
    # kps = gen_keypoints(obj_pose)
    # kps = abs_traj(kps, obj_pose[0])
    # kps = kps.reshape(-1,9)
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection='3d')
    # fig_lims = 1.2
    fig_lims = 0.6

    def animate(args):
        kp = args
        kp = kp.reshape(-1,3)
        ax.cla()
        for i in range(len(kp)):
            ax.scatter(kp[i,0], kp[i,1], kp[i,2], color=plt.cm.rainbow(i/len(kp)))
        ax.set_xlim(-fig_lims, fig_lims)
        ax.set_ylim(-fig_lims, fig_lims)
        ax.set_zlim(-fig_lims, fig_lims)
    
    ani = FuncAnimation(fig, animate, frames=kps, interval=100, save_count=sys.maxsize)
    # ani.save(os.path.join(output_dir,'obsact_org_dataset.mp4'), writer='ffmpeg', fps=10) 
    # ani.save(os.path.join(output_dir,'obsact_org_loader_dataset.mp4'), writer='ffmpeg', fps=10) 
    # ani.save(os.path.join(output_dir,'obsact_abs_dataset.mp4'), writer='ffmpeg', fps=10) 
    ani.save(os.path.join(output_dir,'obsact_abs_loader_dataset.mp4'), writer='ffmpeg', fps=10) 
    plt.show()

    # # create PushT env with keypoints
    # kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    # env = PushTKeypointsEnv(render_size=512, render_action=False, **kp_kwargs)
    # clock = pygame.time.Clock()


    # # get policy from workspace
    # policy = workspace.model
    # if cfg.training.use_ema:
    #     policy = workspace.ema_model
    
    # device = torch.device(device)
    # policy.to(device)
    # policy.eval()

    # # policy
    # obs_in = []
    # act_out = []
    # demo_counts = []
    # env_imgs = []
    # gen_random_traj = False
    # for i in range(tested_demos):
    #     chunks = list(range(ends[i], ends[i+1], 16))
        
    #     for k in range(len(chunks)-1):
    #         demo_counts.append([i]*16)

    #         kp_chunk = kps[chunks[k]:chunks[k+1]] # 16,9,2
    #         center_pos = get_center_pos(kp_chunk[0])
    #         center_ang = get_center_ang(kp_chunk[0])
    #         kp_traj = centralize(kp_chunk, center_pos, center_ang, screen_size)

    #         act_chunk = acts[chunks[k]:chunks[k+1]] # 16,2
    #         init_action = centralize(act_chunk, center_pos, center_ang, screen_size)[0]

    #         if gen_random_traj:
    #             kp_traj = generate_kp_traj(kp_traj,8)
    #             init_action = [np.random.randint(50, 450), np.random.randint(50, 450),]
    #         np_obs_dict = {
    #             'obs': np.expand_dims(kp_traj.reshape(16,18),0).astype(np.float32),
    #             'init_action': np.expand_dims(init_action,0).astype(np.float32),
    #         }
    #         # obs_in.append(kp_traj)
    #         obs_in.append(decentralize(kp_traj, center_pos, center_ang, screen_size))


    #         # device transfer
    #         obs_dict = dict_apply(np_obs_dict, 
    #             lambda x: torch.from_numpy(x).to(
    #                 device=device))

    #         # # run policy
    #         with torch.no_grad():
    #             action_dict = policy.predict_action(obs_dict)

    #         # # device_transfer
    #         np_action_dict = dict_apply(action_dict,
    #             lambda x: x.detach().to('cpu').numpy())
            
    #         np_action = decentralize(np_action_dict['action_pred'].squeeze(0), center_pos, center_ang, screen_size)
    #         act_out.append(np_action)


    #         # reset env and get observations (including info and render for recording)
    #         reset_state = states[chunks[k]]
    #         # reset_state[:4] = centralize(reset_state[:4].reshape(2,2), center_pos, center_ang, screen_size).reshape(4,)
    #         # reset_state[:2] = init_action
    #         # reset_state[4] = 0

    #         env.reset_to_state = reset_state
    #         obs = env.reset()
    #         info = env._get_info()
    #         img = env.render(mode='human')

    #         for act in np_action:
    #             # step env and render
    #             obs, reward, done, info = env.step(act)
    #             img = env.render(mode='human')
    #             env_imgs.append(img)
            
    #         # regulate control frequency
    #         control_hz = 10
    #         clock.tick(control_hz)

        
    # obs_in = np.vstack((obs_in))
    # act_out = np.vstack((act_out))
    # demo_counts = np.array(demo_counts).flatten()
    # ani = FuncAnimation(fig, animate, frames=zip(obs_in,act_out,demo_counts,env_imgs), interval=100, save_count=sys.maxsize)
    # ani.save(os.path.join(output_dir,'obs_to_act_id.mp4'), writer='ffmpeg', fps=10) 
    # plt.show()



if __name__ == '__main__':
    main()
