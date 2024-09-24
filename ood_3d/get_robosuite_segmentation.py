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
from omegaconf import OmegaConf
import torch
import dill
import wandb
import collections
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
import numpy as np
import cv2

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera, get_camera_segmentation, transform_from_pixels_to_world, get_real_depth_map

from utils import to_obj_pose, gen_keypoints, abs_traj, abs_se3_vector, deabs_se3_vector, obs_quat_to_rot6d, quat_correction

from config import combined_policy_cfg


def draw_frame_axis_to_2d(T, ax, world_to_pixel, color, render_size, length=0.05, alpha=1.0):
    if ax is None:
        return
    
    x_axis = T_multi_vec(T, np.array([length,    0,    0]))
    y_axis = T_multi_vec(T, np.array([0,    length,    0]))
    z_axis = T_multi_vec(T, np.array([0,    0,    length]))

    center = T_multi_vec(T, np.array([0.0, 0.0, 0.0]))
    stack_x = np.vstack((center, x_axis))
    stack_y = np.vstack((center, y_axis))
    stack_z = np.vstack((center, z_axis))

    stack_x = project_points_from_world_to_camera(stack_x, 
                                        world_to_camera_transform=world_to_pixel, 
                                        camera_height=render_size, 
                                        camera_width=render_size)
    
    stack_y = project_points_from_world_to_camera(stack_y, 
                                        world_to_camera_transform=world_to_pixel, 
                                        camera_height=render_size, 
                                        camera_width=render_size)
    
    stack_z = project_points_from_world_to_camera(stack_z, 
                                        world_to_camera_transform=world_to_pixel, 
                                        camera_height=render_size, 
                                        camera_width=render_size)

    ax.plot(stack_x[:,1], stack_x[:,0], color=color, alpha=alpha)
    ax.plot(stack_y[:,1], stack_y[:,0], color=color, alpha=alpha)
    ax.plot(stack_z[:,1], stack_z[:,0], color=color, alpha=alpha)

def draw_frame_axis(T, ax, color, length=0.05, alpha=1.0):
    if ax is None:
        return
    
    x_axis = T_multi_vec(T, np.array([length,    0,    0]))
    y_axis = T_multi_vec(T, np.array([0,    length,    0]))
    z_axis = T_multi_vec(T, np.array([0,    0,    length]))

    center = T_multi_vec(T, np.array([0.0, 0.0, 0.0]))
    stack_x = np.vstack((center, x_axis))
    stack_y = np.vstack((center, y_axis))
    stack_z = np.vstack((center, z_axis))

    ax.plot(stack_x[:,0], stack_x[:,1], stack_x[:,2], color=color, alpha=alpha)
    ax.plot(stack_y[:,0], stack_y[:,1], stack_y[:,2], color=color, alpha=alpha)
    ax.plot(stack_z[:,0], stack_z[:,1], stack_z[:,2], color=color, alpha=alpha)

def T_multi_vec(T, vec):
    vec = vec.flatten()
    return (T @ np.append(vec, 1.0).reshape(-1,1)).flatten()[:3]

def check_neighbors(img, values, position, neighbors, threshold):
    x,y = position
    correct_values = 0
    for x_inc in neighbors:
        for y_inc in neighbors:
            if img[x+x_inc, y+y_inc] in values: correct_values += 1
    return correct_values > threshold

def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
        use_depth_obs=True
    )
    return env


@click.command()
@click.option('-d', '--data_path', required=True)
@click.option('-o', '--output_dir', required=True)
def main(data_path, output_dir):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    cfg_path = os.path.expanduser('/home/george/diffusion_policy/diffusion_policy/config/task/square_image_abs.yaml')
    cfg = OmegaConf.load(cfg_path)
    shape_meta = cfg['shape_meta']
    
    dataset_path = os.path.expanduser(data_path)
    dataset = h5py.File(dataset_path,'r')
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)

    ### CAMERA USED ###
    camera = 'highview'

    if cfg.abs_action:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False
    env_meta['env_kwargs']['camera_depths'] = True
    env_meta['env_kwargs']['camera_heights'] = cfg['shape_meta']['obs'][camera+'_image']['shape'][1]
    env_meta['env_kwargs']['camera_widths'] = cfg['shape_meta']['obs'][camera+'_image']['shape'][2]
    env_meta['env_kwargs']['camera_names'] = ['highview']

    env = create_env(env_meta, shape_meta)
    obs = env.reset()

    # plt.imshow(np.moveaxis(obs['highview_image'],0,2))
    # plt.show()
   
    
    world_to_camera_transform = get_camera_transform_matrix(sim=env.env.sim, 
                                                          camera_name=camera, 
                                                          camera_height=cfg['shape_meta']['obs'][camera+'_image']['shape'][1], 
                                                          camera_width=cfg['shape_meta']['obs'][camera+'_image']['shape'][2])
    camera_to_world_transform = np.linalg.inv(world_to_camera_transform)

    
    envs_tested = list(range(10))
    np.random.seed(3501000)
    ood_offsets = np.random.uniform([-0.01,-0.35],[0.01,-0.20],(len(envs_tested),2))
    segmented_ptclouds = []
    n_pts = 600

    for k in range(len(envs_tested)):
        n = envs_tested[k]
        init_state = dataset[f'data/demo_{n}/states'][0]
        # print(dataset[f'data/demo_{n}/states'].shape)
        # print(dataset[f'data/demo_{n}/obs/object'].shape)
        # i=10,11,12 is xyz of object
        init_state[10:12] = init_state[10:12] + ood_offsets[k]
        obs = env.reset_to({'states': init_state})
        # obs = env.reset()

        segmentation = get_camera_segmentation(sim=env.env.sim, 
                                               camera_name=camera, 
                                               camera_height=cfg['shape_meta']['obs'][camera+'_image']['shape'][1], 
                                               camera_width=cfg['shape_meta']['obs'][camera+'_image']['shape'][2])
        # depth_map = get_real_depth_map(sim=env.env.sim,
        #                             depth_map=obs['agentview_depth'])
        # depth_map = np.expand_dims(cv2.resize(np.moveaxis(obs['agentview_depth'],0,2),cfg['shape_meta']['obs']['agentview_image']['shape'][1:3]), 2)
        depth_map = np.moveaxis(obs[camera+'_depth'],0,2)

        n_types = int(np.max(segmentation[...,1]))
        segmentation_img = np.zeros((segmentation.shape[0],segmentation.shape[1],3))
        seg_values = list(range(51,56))
        neighbors = list(range(-3,3))
        segmented_pts = []
        for i in range(segmentation.shape[0]):
            for j in range(segmentation.shape[1]):
                if segmentation[i,j,1] in seg_values:
                    if check_neighbors(segmentation[...,1], seg_values, (i,j), neighbors, threshold=7):
                        segmentation_img[i,j] = plt.cm.rainbow(segmentation[i,j,1]/n_types)[:3]
                        segmented_pts.append([i,j])
                    else: segmentation_img[i,j] = plt.cm.rainbow(0)[:3]
                else:
                    segmentation_img[i,j] = plt.cm.rainbow(0)[:3]
        
        segmented_3dpts = transform_from_pixels_to_world(pixels=np.array(segmented_pts), 
                                                         depth_map=np.repeat(depth_map[None], len(segmented_pts), 0), 
                                                         camera_to_world_transform=camera_to_world_transform)
        picked_pts = np.sort(np.random.choice(np.arange(len(segmented_3dpts)), n_pts))
        segmented_3dpts = segmented_3dpts[picked_pts]

        # fig_lims = 1
        # fig = plt.figure(figsize=(20,10))
        # fig.tight_layout()
        # ax1 = fig.add_subplot(1, 2, 1)
        # ax1.imshow(segmentation_img)
        # ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        # ax2.set_xlim(-fig_lims/2, fig_lims/2)
        # ax2.set_ylim(-fig_lims/2, fig_lims/2)
        # ax2.set_zlim(0, fig_lims)
        # for i in range(0,len(segmented_3dpts)):
        #     ax2.scatter(segmented_3dpts[i,0],segmented_3dpts[i,1],segmented_3dpts[i,2], color=plt.cm.rainbow(i/len(segmented_3dpts)), s=1)
        
        # plt.show()

        segmented_ptclouds.append(segmented_3dpts)

    segmented_ptclouds = np.array(segmented_ptclouds)
    print(segmented_ptclouds.shape)
    np.save('output/segmentation/square.npy', segmented_ptclouds)



if __name__ == '__main__':
    main()
