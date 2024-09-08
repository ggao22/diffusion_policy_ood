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
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robosuite.utils.transform_utils import pose2mat

from utils import to_obj_pose, gen_keypoints, abs_traj, deabs_se3_vector

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
import numpy as np


def make_env(cfg):
    dataset_path = os.path.expanduser(cfg.task.dataset['dataset_path'])
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)
    
    if cfg.task.abs_action:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False

    env_obs_keys = [
            'object', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos']
    
    robomimic_env = create_env(
            env_meta=env_meta, 
            obs_keys=env_obs_keys,
        )
    # hard reset doesn't influence lowdim env
    # robomimic_env.env.hard_reset = False
    env = RobomimicLowdimWrapper(
        env=robomimic_env,
        obs_keys=env_obs_keys,
        init_state=None,
        render_hw=(256,256),
        render_camera_name='agentview',
    )
    return env


def create_env(env_meta, obs_keys):
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        # only way to not show collision geometry
        # is to enable render_offscreen
        # which uses a lot of RAM.
        render_offscreen=False,
        use_image_obs=False, 
    )
    return env



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

    env = make_env(cfg)

    # episode_starts = [0] + list(dataset.replay_buffer.episode_ends[:-1])
    # print(dataset.replay_buffer['obs'][:episode_starts[1]][:,:3])

    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, pin_memory=True)
    dataiter = iter(dataloader)
    vis_batch = next(dataiter) #B,horizon,dict
    skips = 16
    obs = vis_batch['obs'][::skips]
    B,H,D = obs.shape
    obj_kps = obs[:,:,:9].reshape(B*H,-1)
    ee = obs[:,:,9:16].reshape(B*H,-1)

    quat2mat = RotationTransformer(from_rep='quaternion', to_rep='matrix')
    euler2mat = RotationTransformer(from_rep='euler_angles', from_convention='YXZ', to_rep='matrix')
    robot_ori = env.env.env.robots[0].base_ori
    robot_ori = np.array(robot_ori[-1:] + robot_ori[:-1])
    mat_corrected = quat2mat.forward(robot_ori) @ euler2mat.forward(np.array([0, 0, -np.pi/2]))
    robot_ori = quat2mat.inverse(mat_corrected)
    robot_in_world_frame = pose2mat((env.env.env.robots[0].base_pos, robot_ori))
    print(robot_in_world_frame)
    ee = deabs_se3_vector(ee, robot_in_world_frame)


    rotation_transformer = RotationTransformer(
        from_rep='quaternion', to_rep='matrix')
    rot = rotation_transformer.forward(ee[:,3:7])
    pose = np.repeat(np.eye(4)[None],ee.shape[0],0)
    pose[:,:3,:3] = rot
    pose[:,:3,3] = ee[:,:3]
    ee_kps = gen_keypoints(pose).reshape(ee.shape[0],-1)
    points = np.hstack((obj_kps,ee_kps))
    grad = vis_batch['kp_gradient'][::skips,0,:]

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
        kp, grad = args
        kp = kp.reshape(-1,3)
        ax.cla()
        for i in range(len(kp)):
            ax.scatter(kp[i,0], kp[i,1], kp[i,2], color=plt.cm.rainbow(i/len(kp)))
        # obj_kp_center = kp[:3].mean(0)
        # print(obj_kp_center.shape)
        # ax.scatter(obj_kp_center[0], obj_kp_center[1], obj_kp_center[2], color=plt.cm.rainbow(1))
        ax.set_xlim(-fig_lims, fig_lims)
        ax.set_ylim(-fig_lims, fig_lims)
        ax.set_zlim(-fig_lims, fig_lims)
    
    ani = FuncAnimation(fig, animate, frames=zip(points,grad), interval=100, save_count=sys.maxsize)
    # ani.save(os.path.join(output_dir,'obsact_org_dataset.mp4'), writer='ffmpeg', fps=10) 
    # ani.save(os.path.join(output_dir,'obsact_org_loader_dataset.mp4'), writer='ffmpeg', fps=10) 
    # ani.save(os.path.join(output_dir,'obsact_abs_dataset.mp4'), writer='ffmpeg', fps=10) 
    # ani.save(os.path.join(output_dir,'obsact_abs_loader_dataset.mp4'), writer='ffmpeg', fps=10) 
    # ani.save(os.path.join(output_dir,'obsact_orgee_loader_dataset.mp4'), writer='ffmpeg', fps=10) 
    # ani.save(os.path.join(output_dir,'obsact_absee_loader_dataset2.mp4'), writer='ffmpeg', fps=10) 
    ani.save(os.path.join(output_dir,'obsact_grad.mp4'), writer='ffmpeg', fps=10) 
    plt.show()


if __name__ == '__main__':
    main()
