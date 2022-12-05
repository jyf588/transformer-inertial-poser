# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) Stanford University

import argparse
import importlib.util
import shutil
import sys
from typing import Tuple
import os
import numpy as np
import pickle
import pybullet as pb
import pybullet_data
import torch

from fairmotion.ops import conversions
from fairmotion.core.motion import Motion

import bullet_client
import dip_loader
from bullet_agent import SimAgent
from data_utils import get_raw_motion_info_nimble_q_dummy_dq
import constants as cst

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Preprocess DIP and Total Capture data with Real IMUs')
parser.add_argument('--is_dip', action='store_true', help='whether to preprocess DIP or Total Capture data')
parser.add_argument('--data_version_tag', type=str, default="v1",
                    help='')
args = parser.parse_args()

DIP_FORMAT = False      # if True, preprocess into DIP and TransPose data format, not used in this repo.
DIP_DATASET = args.is_dip
TAG = args.data_version_tag

def load_motion_dip(motion_file, _char_info):
    m = dip_loader.load(motion=None,
                        file=motion_file,
                        scale=1.0,
                        load_skel=True,
                        load_motion=True,
                        v_up_skel=_char_info.v_up,
                        v_face_skel=_char_info.v_face,
                        v_up_env=_char_info.v_up_env)
    return m


def load_and_augment_dip_motion(
        char: SimAgent,
        name_gt: str,
        name_imu: str
) -> (Motion, np.ndarray, np.ndarray):

    def load(name):
        if name.endswith("npz"):
            data = np.load(name)
        elif name.endswith("pkl"):
            with open(name, "rb") as f:
                data = pickle.load(f, encoding="latin1")
        else:
            assert False

        return data

    _char_info = char.get_char_info()
    motion = load_motion_dip(name_gt, _char_info)

    data_gt = load(name_gt)
    if name_imu == name_gt:
        data_imu = data_gt
    else:
        data_imu = load(name_imu)

    # DIP data set
    # Note: data_gt and data_imu length off a little bit (e.g. 1 frame)
    if "imu_ori" in data_imu:
        imu_R = np.array(data_imu["imu_ori"])       # (seq_len, 17, 3, 3)
        imu_acc = np.array(data_imu["imu_acc"])     # (seq_len, 17, 3)
    # Total Capture dataset
    # Note: somehow IMU order [11, 12, 7, 8, 0, 2] is different from DIP, which is [7, 8, 11, 12, 0, 2]
    elif "ori" in data_imu:
        imu_R_sub = np.array(data_imu["ori"])       # (seq_len, 6, 3, 3)
        imu_acc_sub = np.array(data_imu["acc"])     # (seq_len, 6, 3)
        imu_R = np.zeros((imu_R_sub.shape[0], 17, 3, 3))
        imu_acc = np.zeros((imu_R_sub.shape[0], 17, 3))
        # (ll, rl, lw, rw, h, r)
        imu_R[:, [11, 12, 7, 8, 0, 2], :, :] = imu_R_sub
        imu_acc[:, [11, 12, 7, 8, 0, 2], :] = imu_acc_sub
        # print(imu_R.shape)
        # print(imu_acc.shape)
    else:
        imu_R = imu_acc = np.array([])

    # print(imu_R.shape)

    # augment root_R to motion
    for pose_id, pose in enumerate(motion.poses):
        belly_R = pose.get_transform(_char_info.bvh_map[_char_info.ROOT], local=False)[:3, :3]
        if "trans" in data_gt:
            p = np.array(data_gt["trans"][pose_id])
            root_R = belly_R
        else:
            root_R = cst.rot_up_R.dot(belly_R)
            p = np.array([0, 0, cst.root_z_offset])
        pose.set_transform(_char_info.bvh_map[_char_info.ROOT], conversions.Rp2T(root_R, p), local=False)

    return motion, imu_R, imu_acc


def fill_in_nan_values(H_ori: np.ndarray, H_acc: np.ndarray) -> (np.ndarray, np.ndarray):
    m_len = H_ori.shape[0]

    mask = np.isnan(np.sum(H_ori.reshape((-1, 6, 9)), axis=2))
    for t in range(m_len):
        for i in range(6):
            if mask[t, i]:
                if t <= 10:
                    H_ori[t, i, :, :] = np.nanmean(H_ori[0:10, i, :, :], axis=0)
                else:
                    H_ori[t, i, :, :] = np.nanmean(H_ori[t - 5:t, i, :, :], axis=0)

    mask = np.isnan(np.sum(H_acc, axis=2))
    for t in range(m_len):
        for i in range(6):
            if mask[t, i]:
                if t <= 10:
                    H_acc[t, i, :] = np.nanmean(H_acc[0:10, i, :], axis=0)
                else:
                    H_acc[t, i, :] = np.nanmean(H_acc[t - 5:t, i, :], axis=0)

    assert np.isfinite(np.sum(H_acc))
    assert np.isfinite(np.sum(H_ori))

    return H_ori, H_acc


def get_real_imu_readings_transpose_and_dip_net_format(
    imu_R_real: np.ndarray,
    imu_acc_real: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:

    # two torch matrices of size (l, 6, 3) and (l, 6, 3, 3)
    # all in global frame

    dip_sensors = [7, 8, 11, 12, 0, 2]

    H_ori = imu_R_real[:, dip_sensors, :, :]
    H_acc = imu_acc_real[:, dip_sensors, :]

    H_ori, H_acc = fill_in_nan_values(H_ori, H_acc)

    H_acc = np.einsum('jk,abk->abj', ROT_MAT_DIP_TP, H_acc)
    H_ori = np.einsum('jk,abki->abji', ROT_MAT_DIP_TP, H_ori)

    return torch.Tensor(H_acc[6:-6]), torch.Tensor(H_ori[6:-6])


def get_real_imu_readings_ours_format_knee(
    imu_R_real: np.ndarray,
    imu_acc_real: np.ndarray,
) -> np.ndarray:
    # a matrix of size l-by-(6*(9+3))

    dip_sensors = [2, 7, 8, 11, 12, 0]
    # with new data format, root, lw, rw, lk, rk, head

    H_ori = imu_R_real[:, dip_sensors, :, :]
    H_acc = imu_acc_real[:, dip_sensors, :]

    H_ori, H_acc = fill_in_nan_values(H_ori, H_acc)

    H_acc = np.einsum('jk,abk->abj', ROT_MAT_OURS, H_acc)
    H_ori = np.einsum('jk,abki->abji', ROT_MAT_OURS, H_ori)

    return np.concatenate((
        H_ori.reshape((-1, 6 * 9)),
        H_acc.reshape((-1, 6 * 3))
    ), axis=1)


def load_and_store(char, motion_name_gt, motion_name_imu, save_name):
    print(motion_name_gt, motion_name_imu)
    print(save_name)
    if os.path.exists(save_name):
        print("already generated")
        return

    # Note: s5_freestyle3 in Total Capture has very different IMU and GT SMPL poses length, ignore
    if "s5/freestyle3" in motion_name_gt:
        return

    motion, imu_R, imu_acc = load_and_augment_dip_motion(
        char, motion_name_gt, motion_name_imu
    )

    if DIP_FORMAT:
        h_acc, h_ori = get_real_imu_readings_transpose_and_dip_net_format(
            imu_R, imu_acc
        )
        print(h_acc.size())
        print(h_ori.size())
        torch.save({'acc': h_acc, 'ori': h_ori}, save_name)
    else:
        h = get_real_imu_readings_ours_format_knee(imu_R, imu_acc)
        print(h.shape)
        qdq = get_raw_motion_info_nimble_q_dummy_dq(char, motion)
        print(qdq.shape)
        with open(save_name, "wb") as handle:
            pickle.dump({"imu": h, "nimble_qdq": qdq}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


def gen_data_all_dip(char, src_dir, save_dir):
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        print("warning: path existed")
    except OSError:
        exit()

    count = 0

    list_dirs = [x[0] for x in os.walk(src_dir)]
    for d in list_dirs:
        with os.scandir(d) as it:
            for entry in it:
                if entry.name.endswith('.pkl'):
                    motion_name = os.path.join(d, entry.name)

                    save_ext = ".pt" if DIP_FORMAT else ".pkl"
                    save_name_local = "dipimu_" + d.rsplit('/', 1)[-1] \
                                      + "_" + entry.name[:-4] + save_ext
                    save_name = save_dir + "/" + save_name_local
                    save_name = save_name.replace(" ", "_")

                    load_and_store(char, motion_name, motion_name, save_name)

                    count += 1

    print("count ", count)


def gen_data_all_tc(char, src_gt_dir, src_imu_dir, save_dir):
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        print("warning: path existed")
    except OSError:
        exit()

    count = 0

    list_dirs = [x[0] for x in os.walk(src_gt_dir)]
    for d in list_dirs:
        with os.scandir(d) as it:
            for entry in it:
                if entry.name.endswith('.npz'):
                    motion_name_gt = os.path.join(d, entry.name)

                    motion_name_imu_local = d.rsplit('/', 1)[-1] + "_" + entry.name[:-10]
                    motion_name_imu = os.path.join(src_imu_dir, motion_name_imu_local + ".pkl")

                    save_ext = ".pt" if DIP_FORMAT else ".pkl"
                    save_name_local = "tcimu_" + motion_name_imu_local + save_ext
                    save_name = save_dir + "/" + save_name_local
                    save_name = save_name.replace(" ", "_")

                    load_and_store(char, motion_name_gt, motion_name_imu, save_name)

                    count += 1

    print("count ", count)


def augment_dip_motion_with_syn_SBP(preprocessed_motion_dir, sbp_dir, motion_w_sbp_dir):
    try:
        os.makedirs(motion_w_sbp_dir)
    except FileExistsError:
        print("warning: path existed")
    except OSError:
        exit()

    count = 0

    with os.scandir(preprocessed_motion_dir) as it:
        for entry in it:
            if entry.name.endswith('.pkl'):
                motion_name = os.path.join(preprocessed_motion_dir, entry.name)
                sbp_name = os.path.join(sbp_dir, entry.name)
                motion_w_sbp_name = os.path.join(motion_w_sbp_dir, entry.name)

                if os.path.exists(motion_w_sbp_name):
                    print("already generated")
                    continue

                with open(motion_name, "rb") as handle:
                    motion = pickle.load(handle)
                    imu = motion['imu']
                    qdq = motion['nimble_qdq']
                with open(sbp_name, "rb") as handle:
                    sbp = pickle.load(handle)
                    c = sbp['constrs']
                with open(motion_w_sbp_name, "wb") as handle:
                    pickle.dump(
                        {"imu": imu, "nimble_qdq": qdq, "constrs": c},
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL
                    )
                count += 1

    print("count ", count)


def copy_train_split(all_dir):
    # copy s_01 to s_08 files to data/preprocessed_DIP_IMU_v0_with_aug_c_train
    # s_09 and s_10 are kept in original folder as test split
    save_dir_train = all_dir + "_train"
    try:
        os.makedirs(save_dir_train)
    except FileExistsError:
        print("warning: path existed")
    except OSError:
        exit()

    count = 0
    with os.scandir(all_dir) as it:
        for entry in it:
            if not entry.name.endswith('.pkl'):
                continue
            if entry.name.startswith('dipimu_s_10') or entry.name.startswith('dipimu_s_09'):
                continue
            shutil.copyfile(all_dir + "/" + entry.name, save_dir_train + "/" + entry.name)
            count += 1

    print("train count ", count)


Mode = pb.DIRECT
pb_client = bullet_client.BulletClient(
    connection_mode=Mode)
pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())
pb_client.resetSimulation()

''' Load Character Info Moudle '''
spec = importlib.util.spec_from_file_location(
    "char_info", "amass_char_info.py")
char_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(char_info)

robot = SimAgent(name='sim_agent_0',
                 pybullet_client=pb_client,
                 model_file="data/amass.urdf",
                 char_info=char_info,
                 ref_scale=1.0,
                 self_collision=False,
                 # actuation=spd,
                 kinematic_only=True,
                 verbose=True)

if DIP_DATASET:
    # DIP dataset
    ROT_MAT_OURS = cst.rot_up_R
    ROT_MAT_DIP_TP = conversions.A2R(np.array([0., 0, 0]))

    if DIP_FORMAT:
        gen_data_all_dip(robot, "data/source/DIP_IMU", "data/preprocessed_DIP_IMU_dip")
    else:
        gen_data_all_dip(robot, "data/source/DIP_IMU", "data/preprocessed_DIP_IMU_" + TAG)
        # augment DIP data with SBP info (C) since part of DIP will be used for training.
        # TotalCapture only for testing
        augment_dip_motion_with_syn_SBP(
            "data/preprocessed_DIP_IMU_" + TAG,
            "data/source/preprocessed_DIP_IMU_c",
            "data/preprocessed_DIP_IMU_" + TAG + "_with_aug_c"
        )
        # copy train split for DIP data
        copy_train_split("data/preprocessed_DIP_IMU_" + TAG + "_with_aug_c")
else:
    # TC dataset
    rot_up_R_tc = conversions.A2R(np.array([np.pi / 2, 0, 0]))
    ROT_MAT_OURS = rot_up_R_tc
    ROT_MAT_DIP_TP = np.linalg.inv(cst.rot_up_R) @ rot_up_R_tc
    # rot_up_R @ (I^-1 @ IMU) = rot_up_R_tc @ (X^-1 @ IMU)
    # X = rot_up_R^-1 @ rot_up_R_tc
    # i.e., ROT_MAT_DIP_TP = conversions.A2R(np.array([0, -np.pi/2, 0])

    if DIP_FORMAT:
        gen_data_all_tc(robot, "data/source/TotalCapture",
                        "data/source/TotalCapture_60FPS_Original/", "data/preprocessed_TotalCapture_dip")
    else:
        gen_data_all_tc(robot, "data/source/TotalCapture",
                        "data/source/TotalCapture_60FPS_Original/", "data/preprocessed_TotalCapture_" + TAG)
