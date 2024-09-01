import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import scipy
    
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

# plotting
def draw_latent(zs, save_path):
    fig = plt.figure(figsize=(18,10))
    fig.tight_layout()
    for o in range(1,4):
        ax = fig.add_subplot(1, 3, o, projection='3d')
        x = zs[:,3*(o-1)+0]
        y = zs[:,3*(o-1)+1]
        z = zs[:,3*(o-1)+2]
        for i in range(len(x)):
            ax.scatter(x[i:i+1], y[i:i+1], z[i:i+1], color=plt.cm.rainbow(i/zs.shape[0]), s=30)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f"Point {o}")
    plt.savefig(save_path)


def draw_ood_latent(zs, ood_zs, save_path, grad_arrows=np.array(None), MVNs=[]):
    fig = plt.figure(figsize=(18,10))
    fig.tight_layout()
    for o in range(1,4):
        ax = fig.add_subplot(1, 3, o, projection='3d')
        x = zs[:,3*(o-1)+0]
        y = zs[:,3*(o-1)+1]
        z = zs[:,3*(o-1)+2]
        ood_x = ood_zs[:,3*(o-1)+0]
        ood_y = ood_zs[:,3*(o-1)+1]
        ood_z = ood_zs[:,3*(o-1)+2]
        ax.scatter(x, y, z, color='tab:blue', s=30)
        ax.scatter(ood_x, ood_y, ood_z, color='tab:red', s=30)
        if grad_arrows.any():
            gx = grad_arrows[:,3*(o-1)]
            gy = grad_arrows[:,3*(o-1)+1]
            gz = grad_arrows[:,3*(o-1)+2]

            gu = grad_arrows[:,3*(o-1)+9+0]
            gv = grad_arrows[:,3*(o-1)+9+1]
            gm = grad_arrows[:,3*(o-1)+9+2]
            ax.quiver(gx, gy, gz, gu, gv, gm, length=0.0005, color='k', alpha=0.6)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f"Point {o}")
    plt.savefig(save_path)




# centering kp, assuming pose are available
def abs_traj(traj, pose_0):
    assert traj.shape[-1] == 3

    ### METHOD 1 ###
    # R = scipy.spatial.transform.Rotation
    # r = R.from_matrix(pose_0[:3,:3])
    # r = r.inv()
    # inv_rot = r.as_matrix()
    # inv_pos = -pose_0[:3,3]

    # for i in range(traj.shape[0]):
    #     kps = traj[i] #n_kp,D_kp
    #     traj[i] = (inv_rot @ (kps + inv_pos).T).T

    ### METHOD 2 ###
    org_shape = traj.shape
    traj = traj.reshape(-1,3) #n_kp,D_kp

    # switch into homogenous coordinates
    homo_coord = np.ones((traj.shape[0],1))
    traj_homo = np.hstack((traj,homo_coord)) #n_kp,D_kp+1

    # centering traj on first pose
    transformed_kps_homo = np.linalg.solve(pose_0, traj_homo.T).T #n_kp,D_kp+1
    traj = transformed_kps_homo[:,:3] #n_kp,D_kp
    traj = traj.reshape(org_shape)

    return traj


def deabs_traj(traj, pose_0):
    assert traj.shape[-1] == 3
    org_shape = traj.shape
    traj = traj.reshape(-1,3) #n_kp,D_kp

    # switch into homogenous coordinates
    homo_coord = np.ones((traj.shape[0],1))
    traj_homo = np.hstack((traj,homo_coord)) #n_kp,D_kp+1

    # decentering traj on first pose
    transformed_kps_homo = (pose_0 @ traj_homo.T).T #n_kp,D_kp+1
    traj = transformed_kps_homo[:,:3] #n_kp,D_kp
    traj = traj.reshape(org_shape)
    return traj


def abs_se3_vector(se3_vector, pose_0):
    # assuming D is position + rotation_6d
    assert se3_vector.shape[-1] == 9
    
    org_shape = se3_vector.shape
    se3_vector = se3_vector.reshape(-1,org_shape[-1])

    # switch full pose matrix
    rotation_transformer = RotationTransformer(from_rep='rotation_6d', to_rep='matrix')

    position = se3_vector[:,:3] 
    rot = se3_vector[:,3:] 
    rot = rotation_transformer.forward(rot)

    position = np.expand_dims(position,2) # N,3,1
    partial_pose = np.concatenate((rot,position),2) # N,3,4
    full_pose = np.concatenate((partial_pose, np.repeat([[[0,0,0,1]]],partial_pose.shape[0],0)),1) # N,4,4
    
    # centering traj on first pose
    transformed_full_pose = np.linalg.solve(pose_0, full_pose)
    transformed_position = transformed_full_pose[:,:3,3]
    transformed_r_matrix = transformed_full_pose[:,:3,:3]

    transformed_rot = rotation_transformer.inverse(transformed_r_matrix)
    
    transformed_se3_vector = np.hstack((transformed_position, transformed_rot))
    transformed_se3_vector = transformed_se3_vector.reshape(org_shape)
    return transformed_se3_vector


def deabs_se3_vector(se3_vector, pose_0):
    # assuming D is position + rotation_6d
    assert se3_vector.shape[-1] == 9
    
    org_shape = se3_vector.shape
    se3_vector = se3_vector.reshape(-1,org_shape[-1])

    # switch full pose matrix
    rotation_transformer = RotationTransformer(from_rep='rotation_6d', to_rep='matrix')

    position = se3_vector[:,:3] 
    rot = se3_vector[:,3:] 
    rot = rotation_transformer.forward(rot)

    position = np.expand_dims(position,2) # N,3,1
    partial_pose = np.concatenate((rot,position),2) # N,3,4
    full_pose = np.concatenate((partial_pose, np.repeat([[[0,0,0,1]]],partial_pose.shape[0],0)),1) # N,4,4
    
    # centering traj on first pose
    transformed_full_pose = pose_0 @ full_pose
    transformed_position = transformed_full_pose[:,:3,3]
    transformed_r_matrix = transformed_full_pose[:,:3,:3]

    transformed_rot = rotation_transformer.inverse(transformed_r_matrix)
    
    transformed_se3_vector = np.hstack((transformed_position, transformed_rot))
    transformed_se3_vector = transformed_se3_vector.reshape(org_shape)
    return transformed_se3_vector


def abs_grad(traj, pose_0):
    assert traj.shape[-1] == 3
    org_shape = traj.shape
    traj = traj.reshape(-1,3).T
    pose_0_inv = np.linalg.inv(pose_0)

    # center grad on first pose
    R = pose_0_inv[:3,:3]
    traj = R @ traj
    traj = traj.reshape(org_shape)
    return traj


def robosuite_data_to_obj_dataset(data):
    # data/demo_0/obs/object
    object_dataset = []
    for demo in data['data'].keys():
        obj_obs = np.array(data['data'][demo]['obs']['object'])
        object_dataset.append(obj_obs)
    object_dataset = np.vstack((object_dataset)) # N,D
    return object_dataset

def to_obj_pose(object_dataset):
    rotation_transformer = RotationTransformer(from_rep='rotation_6d', to_rep='matrix')
    object_rotation = rotation_transformer.forward(object_dataset[:,3:7]) # obj dim: [nut_pos, nut_quat, nut_to_eef_pos, nut_to_eef_quat]
    object_pos = object_dataset[:,:3]

    object_pos = np.expand_dims(object_pos,2)
    partial_pose = np.concatenate((object_rotation,object_pos),2)
    object_pose = np.concatenate((partial_pose, np.repeat([[[0,0,0,1]]],partial_pose.shape[0],0)),1)
    return object_pose


# kp generation from pose
def gen_keypoints(poses, est_obj_size=0.05):
    keypoints = []
    for pose in poses:
        t = pose[:3,3]
        R = pose[:3,:3]

        kp_t = []
        for i in range(3):
            kp = R[:3,i] * est_obj_size + t
            kp_t.append(kp)
        keypoints.append(kp_t)

    return np.array(keypoints)



# data stats utils
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats