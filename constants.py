# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) Stanford University

import numpy as np
from fairmotion.ops import conversions

DT = 1. / 60
acc_fd_N = 4
DT_FIN_ACC = DT * acc_fd_N
ROOT_COM_OFFSET = np.array([0.0, 0.1, -0.1])       # for syn IMU generation, root IMU location
NOMINAL_H = 1.7     # for syn IMU generation
V_THRES = 0.15      # for SBP gen thresholding

# IMU pre-processing
IMU_n_smoooth = 5
ACC_MOVING_AVE_LEN = IMU_n_smoooth * 2 + 1
ACC_SUM_WIN_LEN = 40
ACC_SUM_DOWN_SCALE = 15.0  # to bring to similar scale as acc itself
BIAS_NOISE_ACC = 0.1

rot_up_Q = np.array([0.5, 0.5, 0.5, 0.5])
rot_up_R = conversions.Q2R(rot_up_Q)
root_z_offset = 0.95
n_dofs = 57

MAP_BOUND = 5.0
GRID_SIZE = 0.1
GRID_NUM = int(MAP_BOUND/GRID_SIZE) * 2

# note that our model do not predict toe, wrist, hand joints
# since 6 IMUs cannot provide enough info for these 6 joints.

SMPL_JOINTS = [
    "root",
    "lhip",
    "rhip",
    "lowerback",
    "lknee",
    "rknee",
    "upperback",
    "lankle",
    "rankle",
    "chest",
    "ltoe",
    "rtoe",
    "lowerneck",
    "lclavicle",
    "rclavicle",
    "upperneck",
    "lshoulder",
    "rshoulder",
    "lelbow",
    "relbow",
    "lwrist",
    "rwrist",
    "lhand",
    "rhand",
]

SMPL_JOINT_IDX_MAPPING = {x: i for i, x in enumerate(SMPL_JOINTS)}