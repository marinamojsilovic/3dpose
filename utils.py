import numpy as np
from pathlib import Path
import torch

DATA_DIR = "data/"
DATA_3D  = "data/data_3d_h36m.npz"
DATA_2D  = "data/data_2d_h36m_cpn_ft_h36m_dbb.npz"

TRAIN_SUBJECTS = ['S1', 'S5', 'S6', 'S8']
VAL_SUBJECTS   = ['S7']
TEST_SUBJECTS  = ['S9', 'S11']

H36M_TO_17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
NUMBER_OF_JOINTS = 17

INPUT_HEIGHT = 1002
INPUT_WIDTH  = 1000

JOINT_NAMES = {
    'hip':            0,
    'right hip':      1,
    'right knee':     2,
    'right ankle':    3,
    'left hip':       4,
    'left knee':      5,
    'left ankle':     6,
    'spine':          7,
    'thorax':         8,
    'neck':           9,
    'head':           10,
    'left shoulder':  11,
    'left elbow':     12,
    'left wrist':     13,
    'right shoulder': 14,
    'right elbow':    15,
    'right wrist':    16,
}

# load 2d 3d for given subjects
def load_h36m(path_2d, path_3d, subjects):      
    dict_2d_gt = np.load(path_2d, allow_pickle= True)["positions_2d"].item()
    dict_3d_gt = np.load(path_3d, allow_pickle= True)["positions_3d"].item()

    flat_2d = []
    flat_3d = []

    for s in subjects:
        actions = dict_2d_gt[s].keys()
        for a in actions:  
            #cameras = np.shape(dict_2d_gt[s][a])[0]
            cam = 0
            pose_2d = dict_2d_gt[s][a][cam]
            pose_3d = dict_3d_gt[s][a]
            pose_3d = pose_3d[:, H36M_TO_17, :]

            min_len = min(len(pose_2d), len(pose_3d))
            flat_2d.append(pose_2d[:min_len])
            flat_3d.append(pose_3d[:min_len])

    # mosh everything regardless of action for now
    flat_2d = np.concatenate(flat_2d, axis=0)
    flat_3d = np.concatenate(flat_3d, axis=0)

    return flat_2d, flat_3d

# normalize 2d
def normalize_2d(poses_2d): 
    h_half = INPUT_HEIGHT / 2
    w_half = INPUT_WIDTH / 2

    poses_2d[:, :, 0] = (poses_2d[:, :, 0] - w_half) / w_half
    poses_2d[:, :, 1] = (poses_2d[:, :, 1] - h_half) / h_half
            
    return poses_2d

# center 3d around hip joint
def normalize_3d(poses_3d):                        
    hips_loc = poses_3d[:, JOINT_NAMES['hip'], :]
    poses_3d -= hips_loc[:, np.newaxis, :]

    return poses_3d
   
# fo 3d
def compute_mpjpe(pred, gt):    
    error = np.sqrt(((pred - gt)**2).sum(axis = -1))
    return error

if __name__ == "__main__":
    
    kp_2d, kp_3d = load_h36m(DATA_2D, DATA_3D, TRAIN_SUBJECTS)
    kp_2d = normalize_2d(kp_2d)
    kp_3d = normalize_3d(kp_3d)

    print(compute_mpjpe(kp_3d, kp_3d).sum())