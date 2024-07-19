import zarr
import numpy as np
import cv2


def kp_to_objaction(datapath):
    dataset = zarr.open(datapath,'r')
    kps = np.array(dataset['data']['keypoint'])
    ends = np.array([0]+list(dataset['meta']['episode_ends']))

    kpvs = []
    for i in range(len(ends)-1):
        kp = kps[ends[i]:ends[i+1]] 
        kp = kp.reshape(kp.shape[0],-1)
        kpv = np.diff(kp, axis=0)
        v0 = np.zeros((1,kpv.shape[1]))
        kpvs.append(np.vstack((v0, kpv)))

    kpvs = np.vstack((kpvs))
    return kpvs


if __name__ == "__main__":
    #test
    print(kp_to_objaction(datapath='../data/pusht_demo_50.zarr').shape)