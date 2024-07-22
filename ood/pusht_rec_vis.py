import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import click
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
import pygame

@click.command()
@click.option('-rs', '--render_size', default=96, type=int)
@click.option('-hz', '--control_hz', default=10, type=int)
def main(render_size, control_hz):
    """
    TODO: update this
    Collect demonstration for the Push-T task.
    
    Usage: python demo_pusht.py -o data/pusht_demo.zarr
    
    This script is compatible with both Linux and MacOS.
    Hover mouse close to the blue circle to start.
    Push the T block into the green area. 
    The episode will automatically terminate if the task is succeeded.
    Press "Q" to exit.
    Press "R" to retry.
    Hold "Space" to pause.
    """
    cfg = {
        # main cfgs
        "output_dir": "output",
        "input_size": (96,96),
        "dataname": "pusht_demo_left",
        "action_dim": 18,

        "num_workers": 2,
        "batch_size": 64,

        "num_test_traj": 8,
        "n_components": 5,
        
        "load_encoder": True,
        "encoder_max_epoch": 600,
        "encoder_loss_stop_threshold": 1.5e+3,
        "encoder_lr": 2e-4,
        "test_every": 50,

        # testing cfgs
        'ood_datapath': '/home/george/diffusion_policy/data/pusht_demo_right.zarr',
        'testing_dir': '/home/george/diffusion_policy/ood/output/pusht_demo_left; 07-22-2024_15:30',

        # rec cfg
        "eps": -42,
        "tau": 10,
        "eta": 1.0
    }   

    cfg["datapath"] = f'/home/george/diffusion_policy/data/{cfg["dataname"]}.zarr'

    
    n_episodes = 0

    # create PushT env with keypoints
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsEnv(render_size=render_size, render_action=False,  display_rec=True, rec_cfg=cfg, **kp_kwargs)
    agent = env.teleop_agent()
    clock = pygame.time.Clock()
    
    # episode-level while loop
    while True:
        episode = list()
        # record in seed order, starting with 0
        seed = n_episodes
        print(f'starting seed {seed}')

        # set seed for env
        env.seed(seed)
        
        # reset env and get observations (including info and render for recording)
        obs = env.reset()
        info = env._get_info()
        img = env.render(mode='human')
        
        # loop state
        retry = False
        pause = False
        done = False
        plan_idx = 0
        pygame.display.set_caption(f'plan_idx:{plan_idx}')
        # step-level while loop
        while not done:
            # process keypress events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # hold Space to pause
                        plan_idx += 1
                        pygame.display.set_caption(f'plan_idx:{plan_idx}')
                        pause = True
                    elif event.key == pygame.K_r:
                        # press "R" to retry
                        retry=True
                    elif event.key == pygame.K_q:
                        # press "Q" to exit
                        exit(0)
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        pause = False

            # handle control flow
            if retry:
                break
            if pause:
                continue
            
            # get action from mouse
            # None if mouse is not close to the agent
            act = agent.act(obs)
                
            # step env and render
            obs, reward, done, info = env.step(act)
            img = env.render(mode='human')
            
            # regulate control frequency
            clock.tick(control_hz)

        if not retry:
            print(f'done seed {seed}')
            n_episodes += 1
        else:
            print(f'retry seed {seed}')


if __name__ == "__main__":
    main()
