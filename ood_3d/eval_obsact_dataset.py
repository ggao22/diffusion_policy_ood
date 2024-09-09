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
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

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

    print(dataset.replay_buffer.keys())
    episode_starts = [0] + list(dataset.replay_buffer.episode_ends[:-1])
    # print(dataset.replay_buffer['obs'][:episode_starts[1]][:,:3])

    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True)
    dataiter = iter(dataloader)
    vis_batch = next(dataiter) #B,horizon,dict
    skips = 4
    obj_kps = vis_batch['obs'][::skips]
    ee = vis_batch['action'][::skips]
    obj_kps = obj_kps.reshape(-1,obj_kps.shape[-1])
    ee = ee.reshape(-1,ee.shape[-1])

    rotation_transformer = RotationTransformer(
        from_rep='rotation_6d', to_rep='matrix')
    rot = rotation_transformer.forward(ee[:,3:9])
    pose = np.repeat(np.eye(4)[None],ee.shape[0],0)
    pose[:,:3,:3] = rot
    pose[:,:3,3] = ee[:,:3]
    ee_kps = gen_keypoints(pose).reshape(ee.shape[0],-1)
    print(ee_kps.shape)
    print(obj_kps.shape)
    points = np.hstack((obj_kps,ee_kps))

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
    
    ani = FuncAnimation(fig, animate, frames=points, interval=100, save_count=sys.maxsize)
    # ani.save(os.path.join(output_dir,'obsact_org_dataset.mp4'), writer='ffmpeg', fps=10) 
    # ani.save(os.path.join(output_dir,'obsact_org_loader_dataset.mp4'), writer='ffmpeg', fps=10) 
    # ani.save(os.path.join(output_dir,'obsact_abs_dataset.mp4'), writer='ffmpeg', fps=10) 
    # ani.save(os.path.join(output_dir,'obsact_abs_loader_dataset.mp4'), writer='ffmpeg', fps=10) 
    # ani.save(os.path.join(output_dir,'obsact_orgee_loader_dataset.mp4'), writer='ffmpeg', fps=10) 
    ani.save(os.path.join(output_dir,'obsact_absee_loader_dataset2.mp4'), writer='ffmpeg', fps=10) 
    plt.show()


if __name__ == '__main__':
    main()
