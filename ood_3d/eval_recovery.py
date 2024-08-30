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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
import numpy as np

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from sklearn.mixture import GaussianMixture
from utils import to_obj_pose, gen_keypoints, abs_traj, abs_grad, deabs_traj
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
    

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()


    # read from dataset
    dataset_path = os.path.expanduser(cfg.dataset_path)
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)
    
    if cfg.abs_action:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False

    robomimic_env = create_env(
            env_meta=env_meta, 
            obs_keys=cfg.obs_keys
        )
    # hard reset doesn't influence lowdim env
    # robomimic_env.env.hard_reset = False
    env = RobomimicLowdimWrapper(
        env=robomimic_env,
        obs_keys=cfg.obs_keys,
        init_state=None,
        render_hw=(256,256),
        render_camera_name='agentview',
    )

    gmms = []
    for i in range(3):
        gmms.append(GaussianMixture(n_components=rec_cfg["n_components"]))

    gmms_params = np.load(os.path.join(rec_cfg['testing_dir'], "gmms.npz"), allow_pickle=True)
    for i in range(len(gmms_params)):
        for (key,val) in gmms_params[str(i)][()].items():
            setattr(gmms[i], key, val)

    rec_policy = GMMGradient(gmms_params)

    obs_in = []
    act_out = []
    env_imgs = []
    for n in range(10):
        env.seed(n)
        rec_policy.eta = 1.0
        obs = env.reset()
        img = env.render(mode='human')

        # env policy control
        max_iter = 10
        action_horizon = 16
        for _ in range(max_iter):
            obj_pose = to_obj_pose(obs[None])
            kp = gen_keypoints(obj_pose)

            rec_vec = rec_policy(kp)
            kp_traj = generate_kp_traj(kp, rec_vec, horizon=16, delay=10)

            init_action = centralize(np.expand_dims(info['pos_agent'],0), center_pos, center_ang, screen_size)

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
            np_action = decentralize(np_action, center_pos, center_ang, screen_size)

            # step env and render
            for i in range(action_horizon):
                # step env and render
                act = np_action[i]
                obs, reward, done, info = env.step(act)
                img = env.render(mode='human')
                env_imgs.append(img)

            # regulate control frequency
            control_hz = 10
            clock.tick(control_hz)

    obs_in = np.vstack((obs_in))
    act_out = np.vstack((act_out))
    ani = FuncAnimation(fig, animate, frames=zip(obs_in,act_out,env_imgs), interval=100, save_count=sys.maxsize)
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
