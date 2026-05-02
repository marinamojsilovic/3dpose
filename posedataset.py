import numpy as np
from pathlib import Path
import torch
from utils import *

class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, subjects):
        kp_2d, kp_3d = load_h36m(DATA_2D, DATA_3D, subjects)
        self.data_2d = torch.from_numpy(normalize_2d(kp_2d).astype(np.float32))
        self.data_3d = torch.from_numpy(normalize_3d(kp_3d).astype(np.float32))
        
        
    def __len__(self):
        return len(self.data_2d)

    def __getitem__(self, idx):
        return self.data_2d[idx], self.data_3d[idx]

if __name__ == "__main__":
    dataset = PoseDataset(TRAIN_SUBJECTS)
    #print(dataset[0])
    print(len(dataset))    
    print(dataset.data_2d.shape)
    print(dataset.data_3d.shape)