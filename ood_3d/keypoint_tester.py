import sys

import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  



# kp generation from pose
def gen_keypoints(poses, est_obj_size=0.15):
    keypoints = []
    for pose in poses:
        t = pose[:3,3]
        R = pose[:3,:3]

        kp_t = []
        for i in range(3):
            kp = R[:3,i] * est_obj_size + t
            kp_t.append(kp)
        kp_t.append(t)
        keypoints.append(kp_t)

    return np.array(keypoints)


poses = np.repeat(np.eye(4)[None],80,0)
angle = 0.2
t_inc = 0.01
for i in range(len(poses)-1):
    ori = poses[i][:3,:3]
    trans = np.array([
        [cos(angle), -sin(angle), 0],
        [sin(angle), cos(angle), 0],
        [0,0,1]
    ])
    trans2 = np.array([
        [1,0,0],
        [0, cos(angle), -sin(angle)],
        [0, sin(angle), cos(angle)],
        
    ])
    poses[i+1][:3,:3] = trans2 @ trans @ ori
    poses[i+1][:3,3] = poses[i][:3,3] + t_inc

kps = gen_keypoints(poses)

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(111, projection='3d')
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
ani.save('kp.mp4', writer='ffmpeg', fps=10) 

plt.show()