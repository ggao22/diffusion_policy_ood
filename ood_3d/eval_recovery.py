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
    dataset = h5py.File(cfg.task.dataset['dataset_path'],'r')

    fig, ax = plt.subplots()

    def animate(args):
        env_img = args
        ax.cla()
        ax.imshow(env_img)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # read from dataset
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

    rec_iter = 10
    env_imgs = []
    env_idx = 10
    for n in range(env_idx,env_idx+1):
        env.init_state = dataset[f'data/demo_{n}/states'][0]
        # i=10,11,12 is xyz of object
        env.init_state[11] = env.init_state[11] - 0.20
        obs = env.reset()
        img = env.render(mode='rgb_array')

        rec_policy.eta = 1.0

        # env policy control
        delay = 16
        for i in range(rec_iter):
            cur_obj_pose = to_obj_pose(obs[:7][None])
            cur_kp = gen_keypoints(cur_obj_pose) # 1,n_kp,D_kp
            densities, rec_vectors = rec_policy(cur_kp)
            rec_vectors = rec_vectors.reshape(cur_kp.shape[1:])
            kp_traj = generate_kp_traj(cur_kp[0], rec_vectors, horizon=16, delay=delay, alpha=0.0001) # H,n_kp,D_kp
            delay -= 1
            abs_kp = abs_traj(kp_traj, cur_obj_pose[0])

            cur_quat = np.concatenate((obs[14+6:14+7], obs[14+3:14+6]))
            cur_mat_corrected = quat2mat.forward(cur_quat) @ euler2mat.forward(np.array([0, 0, -np.pi/2]))
            cur_rot6d_corrected = mat2rot6d.forward(cur_mat_corrected)
            cur_se3 = np.concatenate((obs[14:14+3], cur_rot6d_corrected))[None]
            gripper = -1 if sum(np.abs(obs[-2:]))/2 > 0.02 else 1
            cur_action = np.hstack((abs_se3_vector(cur_se3, cur_obj_pose[0]), np.array([[gripper]])))

            np_obs_dict = {
                'obs': abs_kp.reshape(cfg.horizon,-1)[None].astype(np.float32),
                'init_action': cur_action.astype(np.float32)
            }
            
            # device transfer
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).to(
                    device=device))

            # run policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())

            np_action = np_action_dict['action_pred'].squeeze(0)
            detrans_np_action = deabs_se3_vector(np_action[:,:9], cur_obj_pose[0])

            # print('difference',actions-np.hstack((detrans_np_action, np_action[:,9:])))

            detrans_np_action = np.hstack((detrans_np_action[:,:3], 
                                            vec2rot6d.inverse(detrans_np_action[:,3:9]),
                                            np_action[:,9:]))
            
            # step env and render
            for i in range(detrans_np_action.shape[0]):
                act = detrans_np_action[i]
                for _ in range(2):
                    # step env and render
                    obs, reward, done, info = env.step(act)
                img = env.render(mode='rgb_array')
                env_imgs.append(img)


    ani = FuncAnimation(fig, animate, frames=env_imgs, interval=100, save_count=sys.maxsize)
    ani.save(os.path.join(output_dir,'recovery.mp4'), writer='ffmpeg', fps=10) 
    plt.show()


def generate_kp_traj(kp_start, recovery_vec, horizon, delay, alpha=0.01):
    n_kp,d_kp = kp_start.shape
    kp_base = np.repeat([kp_start], horizon, axis=0) # horizon,n_kp,D
    mean_recovery_vec = recovery_vec.mean(axis=0) * alpha
    motion_vecs = np.repeat([mean_recovery_vec], horizon-delay, axis=0) 
    motion_vecs = np.vstack((np.zeros((delay, d_kp)),motion_vecs)) # horizon,D
    vecs = np.repeat(np.cumsum(motion_vecs, axis=0), n_kp, axis=0).reshape(horizon, n_kp, d_kp)
    return kp_base + vecs


if __name__ == '__main__':
    main()
