import torch
from torchvision import transforms
from torch.utils.data import Dataset

import zarr
import numpy as np

from kp_to_objaction import kp_to_objaction


class ObsActPosPairs(Dataset):
    def __init__(self, datapath=None,
                 preprocess=transforms.Compose([transforms.ToPILImage(),
                                                    transforms.ToTensor()])):
        super().__init__()
        assert(datapath is not None)
        data = zarr.open(datapath,'r')
        self.image_dataset = np.array(data['data']['img'])[:,:,:,[2,1,0]]
        self.action_dataset = kp_to_objaction(datapath)
        self.position_dataset = np.array(data['data']['keypoint'])
        self.end_indices = np.array(data['meta']['episode_ends']).astype(int)-1
        self.transforms = preprocess

    def __getitem__(self, index):
        if index not in self.end_indices:
            q1 = self.transforms(self.image_dataset[index])
            q2 = self.transforms(self.image_dataset[index + 1])
            p1 = self.position_dataset[index].flatten()
            p2 = self.position_dataset[index + 1].flatten()
        else:
            q1 = self.transforms(self.image_dataset[index - 1])
            q2 = self.transforms(self.image_dataset[index])
            p1 = self.position_dataset[index - 1].flatten()
            p2 = self.position_dataset[index].flatten()

        action = self.action_dataset[index]
        return q1, q2, p1, p2, action

    def __len__(self):
        return len(self.image_dataset)



class ObsActPairs(Dataset):
    def __init__(self, datapath=None,
                 preprocess=transforms.Compose([transforms.ToPILImage(),
                                                    transforms.ToTensor()])):
        super().__init__()
        assert(datapath is not None)
        data = zarr.open(datapath,'r')
        self.image_dataset = np.array(data['data']['img'])[:,:,:,[2,1,0]]
        self.action_dataset = kp_to_objaction(datapath)
        self.end_indices = np.array(data['meta']['episode_ends']).astype(int)-1
        self.transforms = preprocess

    def __getitem__(self, index):
        if index not in self.end_indices:
            q1 = self.transforms(self.image_dataset[index])
            q2 = self.transforms(self.image_dataset[index + 1])
        else:
            q1 = self.transforms(self.image_dataset[index - 1])
            q2 = self.transforms(self.image_dataset[index])

        action = self.action_dataset[index]
        return q1, q2, action

    def __len__(self):
        return len(self.image_dataset)
    

class ObsPosPairs(Dataset):
    def __init__(self, datapath=None,
                 preprocess=transforms.Compose([transforms.ToPILImage(),
                                                    transforms.ToTensor()])):
        super().__init__()
        assert(datapath is not None)
        data = zarr.open(datapath,'r')
        self.image_dataset = np.array(data['data']['img'])[:,:,:,[2,1,0]]
        self.position_dataset = np.array(data['data']['keypoint'])
        self.end_indices = np.array(data['meta']['episode_ends']).astype(int)-1
        self.transforms = preprocess

    def __getitem__(self, index):
        img = self.transforms(self.image_dataset[index])
        pos = self.position_dataset[index].flatten()
        return img, pos

    def __len__(self):
        return len(self.image_dataset)


