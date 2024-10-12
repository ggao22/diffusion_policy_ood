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
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
import numpy as np
import collections
import yaml


from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robosuite.utils.transform_utils import pose2mat
from camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera

from utils import to_obj_pose, gen_keypoints, abs_traj, abs_se3_vector, deabs_se3_vector, obs_quat_to_rot6d, quat_correction

from sklearn.mixture import GaussianMixture
from models import GMMGradient
from config import cfg as rec_cfg
from config import combined_policy_cfg

RENDER_SIZE = 84

def draw_frame_axis_to_2d(T, ax, world_to_pixel, color, length=0.05, alpha=1.0):
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
                                        camera_height=RENDER_SIZE, 
                                        camera_width=RENDER_SIZE)
    
    stack_y = project_points_from_world_to_camera(stack_y, 
                                        world_to_camera_transform=world_to_pixel, 
                                        camera_height=RENDER_SIZE, 
                                        camera_width=RENDER_SIZE)
    
    stack_z = project_points_from_world_to_camera(stack_z, 
                                        world_to_camera_transform=world_to_pixel, 
                                        camera_height=RENDER_SIZE, 
                                        camera_width=RENDER_SIZE)

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

def plot_pose(pose, dense=False):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_aspect('equal')
    draw_frame_axis(pose, ax, 'green', 0.08)
    plt.show()


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



def make_env(cfg):
    dataset_path = os.path.expanduser(cfg.task.dataset['dataset_path'])
    # read from dataset
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)
    env_meta['env_kwargs']['use_object_obs'] = True

    if cfg.task.abs_action:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False

    shape_meta = yaml.safe_load("""
    obs:
      agentview_image:
        shape: [3, 84, 84]
        type: rgb
      robot0_eye_in_hand_image:
        shape: [3, 84, 84]
        type: rgb
      robot0_eef_pos:
        shape: [3]
      robot0_eef_quat:
        shape: [4]
      robot0_gripper_qpos:
        shape: [2]
      object:
        shape: [14]
    action: 
        shape: [10]
    """)
    
    robomimic_env = create_env(
                env_meta=env_meta, 
                shape_meta=shape_meta
            )
    # hard reset doesn't influence lowdim env
    # robomimic_env.env.hard_reset = False
    env = RobomimicImageWrapper(
        env=robomimic_env,
        shape_meta=shape_meta,
        init_state=None,
        render_obs_key='agentview_image'
    )
    return env

def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        # only way to not show collision geometry
        # is to enable render_offscreen
        # which uses a lot of RAM.
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
    )
    return env

def add_obs(new_obs_dict, past_obs_dict, n_obs_steps):
    for key in new_obs_dict.keys():
        if key=='object': continue
        new_obs = new_obs_dict[key][None]
        past_obs = past_obs_dict[key][0]
        if len(past_obs) < 1:
            past_obs = np.repeat(new_obs, n_obs_steps, 0)
        else:
            old_obs = past_obs[:-1]
            past_obs = np.vstack((new_obs,old_obs))
        if key!='agentview_image': 
            past_obs = past_obs.astype(np.float32)
        past_obs_dict[key] = past_obs[None]
    return past_obs_dict


def generate_kp_traj(kp_start, recovery_vec, horizon, delay, alpha=0.01):
    n_kp,d_kp = kp_start.shape
    kp_base = np.repeat([kp_start], horizon, axis=0) # horizon,n_kp,D
    mean_recovery_vec = recovery_vec.mean(axis=0) * alpha
    motion_vecs = np.repeat([mean_recovery_vec], horizon-delay, axis=0) 
    motion_vecs = np.vstack((np.zeros((delay, d_kp)),motion_vecs)) # horizon,D
    vecs = np.repeat(np.cumsum(motion_vecs, axis=0), n_kp, axis=0).reshape(horizon, n_kp, d_kp)
    return kp_base + vecs

def generate_recovery_kp_traj(obs, rec_policy, horizon, delay, alpha=0.01, random_walk=0.0):
    cur_obj_pose = to_obj_pose(obs[:7][None])
    cur_kp = gen_keypoints(cur_obj_pose) # 1,n_kp,D_kp
    kp_base = np.repeat(cur_kp, horizon, axis=0) # horizon,n_kp,D_kp
    _,n_kp,d_kp = cur_kp.shape

    kp_motion = np.zeros((delay, d_kp))
    for i in range(horizon-delay):
        densities, rec_vectors = rec_policy(cur_kp)
        rec_vectors = rec_vectors.reshape(cur_kp.shape[1:])
        kp_inc = rec_vectors.mean(axis=0) * alpha + np.random.randn(3)*random_walk
        kp_motion = np.vstack((kp_motion,[kp_inc]))
        cur_kp[0] = cur_kp[0] + np.repeat([kp_inc],3,0)
    kp_motion = np.repeat(np.cumsum(kp_motion, axis=0), n_kp, axis=0).reshape(horizon, n_kp, d_kp)
    return kp_base + kp_motion

def update_delay(pos_agent,pos_object,horizon=16,min_dist=0.165,max_dist=0.22):
    x = np.linalg.norm(pos_agent-pos_object)
    sub_delay = int((-horizon/(max_dist-min_dist)) * (x-min_dist) + horizon)
    delay = min(max(0,horizon-sub_delay),horizon)
    return delay


@click.command()
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load translator policy from checkpoint
    translator_policy, translator_cfg = load_policy(combined_policy_cfg['recovery_ckpt'], device, output_dir)
    # load base policy from checkpoint
    base_policy, base_cfg = load_policy(combined_policy_cfg['base_ckpt'], device, output_dir)

    base_cfg.task.dataset['dataset_path'] = '../' + base_cfg.task.dataset['dataset_path']
    dataset = h5py.File(base_cfg.task.dataset['dataset_path'],'r')

    env = make_env(base_cfg)
    camera_transform_matrix = get_camera_transform_matrix(sim=env.env.env.sim, 
                                                          camera_name='agentview', 
                                                          camera_height=RENDER_SIZE, 
                                                          camera_width=RENDER_SIZE)

    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(1, 1, 1)
    def animate(args):
        env_img, kp, gripper, env_num, poses, delay = args
        kp = kp.reshape(-1,3)
        ax1.cla()
        ax1.imshow(env_img)
        ax1.set_title(f"Env #{str(env_num)}, Delay {str(delay)}")
        cam_kp = project_points_from_world_to_camera(kp, 
                                                     world_to_camera_transform=camera_transform_matrix, 
                                                     camera_height=RENDER_SIZE, 
                                                     camera_width=RENDER_SIZE)
        for i in range(len(cam_kp)):
            ax1.scatter(cam_kp[i,1], cam_kp[i,0], color=plt.cm.rainbow(i/len(cam_kp)), s=15)
        for pose in poses:
            draw_frame_axis_to_2d(pose, ax1, camera_transform_matrix, color=plt.cm.rainbow(1), length=0.1, alpha=1.0)


    vec2rot6d = RotationTransformer(from_rep='axis_angle', to_rep='rotation_6d')
    vec2mat = RotationTransformer(from_rep='axis_angle', to_rep='matrix')

    gmms = []
    for i in range(3):
        gmms.append(GaussianMixture(n_components=rec_cfg["n_components"]))

    gmms_params = np.load(os.path.join(rec_cfg['testing_dir'], "gmms.npz"), allow_pickle=True)
    for i in range(len(gmms_params)):
        for (key,val) in gmms_params[str(i)][()].items():
            setattr(gmms[i], key, val)

    rec_policy = GMMGradient(gmms_params)

   
    env_imgs = []
    max_iter = 35
    n_obs_steps = base_cfg.n_obs_steps
    # envs_tested = [4,5]
    envs_tested = list(range(1))
    np.random.seed(350)
    ood_offsets = np.random.uniform([-0.01,-0.35],[0.01,-0.25],(len(envs_tested),2))
    env_labels = []
    rewards = []
    kp_vis = []
    poses = []
    gripper_states = []
    delays = []
    OOD_THRESHOLD = 40
    action_horizon = 12
    

    for k in range(len(envs_tested)):
        n = envs_tested[k]
        env.init_state = dataset[f'data/demo_{n}/states'][0]
        # i=10,11,12 is xyz of object
        # env.init_state[10:12] = env.init_state[10:12] + ood_offsets[k]
        obs = env.reset()

        past_obs = {
            'agentview_image': [[]],
            'robot0_eye_in_hand_image': [[]],
            'robot0_eef_pos': [[]],
            'robot0_eef_quat': [[]],
            'robot0_gripper_qpos': [[]],
        }

        past_obs = add_obs(obs, past_obs, n_obs_steps)
        delay = 16
        gripper = -1


        # env policy control
        for iter in range(max_iter):
            cur_obj_pose = to_obj_pose(obs['object'][:7][None])
            cur_kp = gen_keypoints(cur_obj_pose) # 1,n_kp,D_kp
            densities, rec_vectors = rec_policy(cur_kp)
            print(np.mean(densities))
            
            if np.mean(densities) < OOD_THRESHOLD:
                ### Case: ODD
                # rec_vectors = rec_vectors.reshape(cur_kp.shape[1:])
                # kp_traj = generate_kp_traj(cur_kp[0], rec_vectors, horizon=16, delay=delay, alpha=0.0001) # H,n_kp,D_kp
                kp_traj = generate_recovery_kp_traj(obs['object'], rec_policy, horizon=16, delay=delay, alpha=0.00005, random_walk=0.008)
                if delay > 0: delay -= 1
                # delay = update_delay(obs['robot0_eef_pos'], obs['object'][:3], horizon=16, min_dist=0.0612, max_dist=0.077)
                abs_kp = abs_traj(kp_traj, cur_obj_pose[0])

                cur_rot6d = obs_quat_to_rot6d(obs['robot0_eef_quat'])
                cur_se3 = np.concatenate((obs['robot0_eef_pos'], cur_rot6d))[None]
                cur_action = np.hstack((abs_se3_vector(cur_se3, cur_obj_pose[0]), np.array([[gripper]])))

                np_obs_dict = {
                    'obs': abs_kp.reshape(translator_cfg.horizon,-1)[None].astype(np.float32),
                    'init_action': cur_action.astype(np.float32)
                }
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = translator_policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                np_action = np_action_dict['action_pred'].squeeze(0)
                detrans_np_action = deabs_se3_vector(np_action[:,:9], cur_obj_pose[0])
                detrans_np_action = np.hstack((detrans_np_action[:,:3], 
                                                vec2rot6d.inverse(detrans_np_action[:,3:9]),
                                                np_action[:,9:]))

                #for visualization
                kp_traj = kp_traj[:action_horizon].reshape(-1,9)
                mat = vec2mat.forward(detrans_np_action[:action_horizon,3:6])
                pose = np.repeat(np.eye(4)[None],action_horizon,0)
                pose[:,:3,:3] = mat
                pose[:,:3,3] = detrans_np_action[:action_horizon,:3]
                ee_kps = gen_keypoints(pose).reshape(action_horizon,-1)
                kp = np.hstack((kp_traj,ee_kps))
                kp_vis.append(kp)
                
                # step env and render
                # detrans_np_action.shape[0]
                for i in range(action_horizon):
                    act = detrans_np_action[i]
                    gripper = act[-1]
                    for _ in range(10):
                        # step env and render
                        obs, reward, done, info = env.step(act)
                    past_obs = add_obs(obs, past_obs, n_obs_steps)
                    img = env.render(mode='rgb_array')
                    env_imgs.append(img)
                    env_labels.append(n)
                    gripper_states.append(gripper)
                    delays.append(delay)
                    poses.append(
                        [to_obj_pose(obs['object'][:7][None])[0], 
                         to_obj_pose(np.concatenate((obs['robot0_eef_pos'], 
                                                     quat_correction(obs['robot0_eef_quat'])))[None])[0]])

            else:
                ## Case: ID
                np_obs_dict = past_obs.copy()

                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = base_policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)
                action = np.hstack((action[:,:3], 
                                vec2rot6d.inverse(action[:,3:9]),
                                action[:,9:]))
                
                # step env and render
                for i in range(action.shape[0]):
                    act = action[i]
                    gripper = act[-1]
                    obs, reward, done, info = env.step(act)
                    past_obs = add_obs(obs, past_obs, n_obs_steps)
                    img = env.render(mode='rgb_array')
                    env_imgs.append(img)
                    env_labels.append(n)
                    gripper_states.append(gripper)
                    delays.append(delay)
                    poses.append(
                        [to_obj_pose(obs['object'][:7][None])[0], 
                         to_obj_pose(np.concatenate((obs['robot0_eef_pos'], 
                                                     quat_correction(obs['robot0_eef_quat'])))[None])[0]])
                    
                    if reward > 0.98: done = True
                    if done: break
                kp_vis.append(np.zeros((i+1,18)))
                if done: break
            
        rewards.append(reward)
        if done:
            print(f'Env #{n} done')
        else:
            print(f'Env #{n} failed, max iter reached. Reward Managed {reward}')

    print(f"Test done, average reward {np.mean(rewards)}")
    kp_vis = np.vstack((kp_vis))
    ani = FuncAnimation(fig, animate, frames=zip(env_imgs,kp_vis,gripper_states,env_labels,poses,delays), interval=100, save_count=sys.maxsize)
    ani.save(os.path.join(output_dir,'combined_vision_ood.mp4'), writer='ffmpeg', fps=10) 
    plt.show()





if __name__ == '__main__':
    main()
