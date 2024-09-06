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

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from utils import to_obj_pose, gen_keypoints, abs_traj, abs_se3_vector, deabs_se3_vector

from sklearn.mixture import GaussianMixture
from models import GMMGradient
from config import cfg as rec_cfg
from config import combined_policy_cfg


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


def add_obs(new_obs, past_obs, n_obs_steps):
    new_obs = new_obs[None]
    if len(past_obs) < 1:
        past_obs = np.repeat(new_obs, n_obs_steps, 0)
    else:
        old_obs = past_obs[:-1]
        past_obs = np.vstack((new_obs,old_obs))
    return past_obs


def generate_kp_traj(kp_start, recovery_vec, horizon, delay, alpha=0.01):
    n_kp,d_kp = kp_start.shape
    kp_base = np.repeat([kp_start], horizon, axis=0) # horizon,n_kp,D
    mean_recovery_vec = recovery_vec.mean(axis=0) * alpha
    motion_vecs = np.repeat([mean_recovery_vec], horizon-delay, axis=0) 
    motion_vecs = np.vstack((np.zeros((delay, d_kp)),motion_vecs)) # horizon,D
    vecs = np.repeat(np.cumsum(motion_vecs, axis=0), n_kp, axis=0).reshape(horizon, n_kp, d_kp)
    return kp_base + vecs



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

    fig, ax = plt.subplots()
    def animate(args):
        env_img, env_num = args
        ax.cla()
        ax.imshow(env_img)
        ax.set_title(f"Env #{str(env_num)}")

    env = make_env(base_cfg)

    vec2rot6d = RotationTransformer(from_rep='axis_angle', to_rep='rotation_6d')
    quat2mat = RotationTransformer(from_rep='quaternion', to_rep='matrix')
    mat2rot6d = RotationTransformer(from_rep='matrix', to_rep='rotation_6d')
    euler2mat = RotationTransformer(from_rep='euler_angles', from_convention='YXZ', to_rep='matrix')

    gmms = []
    for i in range(3):
        gmms.append(GaussianMixture(n_components=rec_cfg["n_components"]))

    gmms_params = np.load(os.path.join(rec_cfg['testing_dir'], "gmms.npz"), allow_pickle=True)
    for i in range(len(gmms_params)):
        for (key,val) in gmms_params[str(i)][()].items():
            setattr(gmms[i], key, val)

    rec_policy = GMMGradient(gmms_params)

   
    env_imgs = []
    max_iter = 50
    n_obs_steps = base_cfg.n_obs_steps
    # envs_tested = [0,2,4,5,6]
    envs_tested = list(range(10))
    np.random.seed(0)
    ood_offsets = np.random.uniform([-0.01,-0.35],[0.01,-0.20],(len(envs_tested),2))
    env_labels = []
    rewards = []
    OOD_THRESHOLD = 0.15
    

    for k in range(len(envs_tested)):
        n = envs_tested[k]
        env.init_state = dataset[f'data/demo_{n}/states'][0]
        # i=10,11,12 is xyz of object
        env.init_state[10:12] = env.init_state[10:12] + ood_offsets[k]
        obs = env.reset()

        past_obs = []
        past_obs = add_obs(obs, past_obs, n_obs_steps)
        rec_policy.eta = 1.0
        delay = 16
        gripper = -1

        # env policy control
        for iter in range(max_iter):

            cur_obj_pose = to_obj_pose(obs[:7][None])
            cur_kp = gen_keypoints(cur_obj_pose) # 1,n_kp,D_kp
            densities, rec_vectors = rec_policy(cur_kp)
            # print(np.mean(densities))
            
            if np.mean(densities) < OOD_THRESHOLD:
                ### Case: ODD
                rec_vectors = rec_vectors.reshape(cur_kp.shape[1:])
                kp_traj = generate_kp_traj(cur_kp[0], rec_vectors, horizon=16, delay=delay, alpha=0.0001) # H,n_kp,D_kp
                if delay > 4: delay -= 1
                abs_kp = abs_traj(kp_traj, cur_obj_pose[0])

                cur_quat = np.concatenate((obs[14+6:14+7], obs[14+3:14+6]))
                cur_mat_corrected = quat2mat.forward(cur_quat) @ euler2mat.forward(np.array([0, 0, -np.pi/2]))
                cur_rot6d_corrected = mat2rot6d.forward(cur_mat_corrected)
                cur_se3 = np.concatenate((obs[14:14+3], cur_rot6d_corrected))[None]
                # gripper = -1 if sum(np.abs(obs[-2:]))/2 > 0.018 else 1
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
                
                # step env and render
                # detrans_np_action.shape[0]
                for i in range(8):
                    act = detrans_np_action[i]
                    gripper = act[-1]
                    for _ in range(3):
                        # step env and render
                        obs, reward, done, info = env.step(act)
                    past_obs = add_obs(obs, past_obs, n_obs_steps)
                    img = env.render(mode='rgb_array')
                    env_imgs.append(img)
                    env_labels.append(n)

            else:
                ## Case: ID
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': past_obs[None].astype(np.float32)
                }
                
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

                    if reward > 0.98: done = True
                    if done: break
                if done: break
        rewards.append(reward)
        if done:
            print(f'Env #{n} done')
        else:
            print(f'Env #{n} failed, max iter reached. Reward Managed {reward}')

    print(f"Test done, average reward {np.mean(rewards)}")
    ani = FuncAnimation(fig, animate, frames=zip(env_imgs,env_labels), interval=100, save_count=sys.maxsize)
    ani.save(os.path.join(output_dir,'combined_ood.mp4'), writer='ffmpeg', fps=10) 
    plt.show()





if __name__ == '__main__':
    main()
