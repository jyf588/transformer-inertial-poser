# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) Stanford University
# untested helper script for your convenience

import importlib.util
import pybullet_data
import scipy
import sys
import pybullet as pb
import numpy as np

from fairmotion.ops import conversions
from fairmotion.core.motion import Motion

import bullet_client
from data_utils import draw_ori
from preprocess_DIP_TC_new import load_and_augment_dip_motion
from bullet_agent import SimAgent
import constants as cst

DT = 1. / 60
acc_fd_N = 4
DT_FIN_ACC = DT * acc_fd_N
WIN_LEN = 11
NOMINAL_H = 1.7
rot_up_R = cst.rot_up_R

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

motion_name_imu = 'data/TotalCapture_60FPS_Original/s1_freestyle3.pkl'
motion_name_gt = 'data/TotalCapture/s1/freestyle3_poses.npz'
# motion_name_gt = motion_name_imu = 'data/DIP_IMU/s_01/03.pkl'
# motion_name_gt = motion_name_imu = 'data/DIP_IMU/s_10/05.pkl'

ROOT_COM_OFFSET = np.array([0.0, 0.1, -0.1])        # 0.1m up, -0.1m back
# 1 TC, 0 DIP
if 1:
    ROT_MAT = conversions.A2R(np.array([np.pi / 2, 0, 0]))
else:
    ROT_MAT = rot_up_R


def viz_motion_and_imu_dip(m: Motion, imu_R: np.ndarray, imu_acc: np.ndarray):

    cur_time = 0.0 / 2.0

    imu_idx = 0
    #
    # link_ind = 11  # lknee
    # link_ind_pb = char_info.lknee

    # link_ind = 12  # rknee
    # link_ind_pb = char_info.rknee

    # link_ind = 8  # rwrist
    # link_ind_pb = char_info.rwrist

    link_ind = 7  # lwrist
    link_ind_pb = char_info.lwrist

    # link_ind = 0  # head
    # link_ind_pb = char_info.upperneck

    plot_log_real = []
    plot_log_syn = []

    while cur_time < 20.0:      # test half

        cur_pose = m.get_pose_by_time(cur_time)
        robot.set_pose(cur_pose, None)

        ###########################################
        real_imu_belly_a = [0., 0, 0]
        imu_idx_prev = max(imu_idx - WIN_LEN//2, 0)
        real_imu_belly_a_s = imu_acc[imu_idx_prev:imu_idx + (WIN_LEN//2+1), 2, :]
        if np.all(np.isfinite(np.nanmean(real_imu_belly_a_s, axis=0))):
            real_imu_belly_a = np.nanmean(real_imu_belly_a_s, axis=0)

        # real_imu_belly_a = imu_acc[imu_idx, 2, :]  # （3，） 2->belly

        real_imu_belly_a = ROT_MAT.dot(real_imu_belly_a)

        cur_xyz = robot.get_root_local_point_p(ROOT_COM_OFFSET)

        prev_pose = m.get_pose_by_time(cur_time - DT_FIN_ACC)
        robot2.set_pose(prev_pose, None)
        prev_xyz = robot2.get_root_local_point_p(ROOT_COM_OFFSET)

        next_pose = m.get_pose_by_time(cur_time + DT_FIN_ACC)
        robot3.set_pose(next_pose, None)
        next_xyz = robot3.get_root_local_point_p(ROOT_COM_OFFSET)

        pb_belly_a = -2 * cur_xyz + prev_xyz + next_xyz
        pb_belly_a /= (DT_FIN_ACC ** 2)
        #
        # pb_client.addUserDebugLine((0, 0, 0), list(pb_belly_a), (0, 1, 0), 5)
        # pb_client.addUserDebugLine((0, 0, 0), list(real_imu_belly_a), (1, 0, 0), 5)

        # all other five acc
        real_imu_j_a = [0., 0, 0]
        imu_idx_prev = max(imu_idx - WIN_LEN//2, 0)
        real_imu_j_a_s = imu_acc[imu_idx_prev:imu_idx + (WIN_LEN//2+1), link_ind, :]
        if np.all(np.isfinite(np.nanmean(real_imu_j_a_s, axis=0))):
            real_imu_j_a = np.nanmean(real_imu_j_a_s, axis=0)

        # real_imu_j_a = imu_acc[imu_idx, link_ind, :]

        real_imu_j_a = ROT_MAT.dot(real_imu_j_a)

        func = robot.get_link_states
        func2 = robot2.get_link_states
        func3 = robot3.get_link_states
        cur_xyz_l = func([link_ind_pb])[0]

        prev_pose = m.get_pose_by_time(cur_time - DT_FIN_ACC)
        robot2.set_pose(prev_pose, None)
        prev_xyz_l = func2([link_ind_pb])[0]

        next_pose = m.get_pose_by_time(cur_time + DT_FIN_ACC)
        robot3.set_pose(next_pose, None)
        next_xyz_l = func3([link_ind_pb])[0]

        pb_j_a = -2 * cur_xyz_l + prev_xyz_l + next_xyz_l
        pb_j_a /= (DT_FIN_ACC ** 2)

        # pb_client.addUserDebugLine((0, 0, 0), list(pb_j_a), (0, 1, 0), 5)
        # pb_client.addUserDebugLine((0, 0, 0), list(real_imu_j_a), (1, 0, 0), 5)

        # plot_log_real.append(real_imu_belly_a)
        # plot_log_syn.append(pb_belly_a)
        plot_log_real.append(real_imu_j_a - real_imu_belly_a)
        plot_log_syn.append(pb_j_a - pb_belly_a)

        #############################
        # # root offset
        # real_imu_belly_R = imu_R[imu_idx, 2, :, :]  # (3,3)
        # real_imu_belly_R = ROT_MAT.dot(real_imu_belly_R)
        #
        # _, pb_belly_q, _, _ = robot.get_root_state()
        # pb_belly_R = conversions.Q2R(pb_belly_q)
        #
        # plot_log_real.append(np.diag(real_imu_belly_R))
        # plot_log_syn.append(np.diag(pb_belly_R))
        #
        # draw_ori(pb_client, real_imu_belly_R)
        # draw_ori(pb_client, pb_belly_R)
        #
        # any other link among the five IMUs
        real_imu_j_R = imu_R[imu_idx, link_ind, :, :]
        real_imu_j_R = ROT_MAT.dot(real_imu_j_R)
        _, pb_j_q, _, _ = robot.get_link_states_joint_frame([link_ind_pb])
        pb_j_R = conversions.Q2R(pb_j_q)

        # plot_log_real.append(np.diag(real_imu_j_R))
        # plot_log_syn.append(np.diag(pb_j_R))

        # draw_ori(pb_client, real_imu_j_R)
        # draw_ori(pb_client, pb_j_R)

        # pb_client.removeAllUserDebugItems()

        cur_time += DT
        imu_idx += 1

    plot_log_real = np.array(plot_log_real)
    plot_log_syn = np.array(plot_log_syn)
    plot_log_syn = scipy.ndimage.uniform_filter1d(plot_log_syn, WIN_LEN, axis=0)
    from matplotlib import pyplot as plt
    plt.subplot(3, 1, 1)
    plt.plot(plot_log_real[:, 0])
    plt.plot(plot_log_syn[:, 0])

    plt.subplot(3, 1, 2)
    plt.plot(plot_log_real[:, 1])
    plt.plot(plot_log_syn[:, 1])

    plt.subplot(3, 1, 3)
    plt.plot(plot_log_real[:, 2])
    plt.plot(plot_log_syn[:, 2])
    plt.show()

    return


Mode = pb.GUI
pb_client = bullet_client.BulletClient(
    connection_mode=Mode)
pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())
pb_client.resetSimulation()

dt_sim = 1. / 60.

''' Load Character Info Moudle '''
spec = importlib.util.spec_from_file_location(
    "char_info", "amass_char_info.py")
char_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(char_info)


robot = SimAgent(name='sim_agent_0',
                           pybullet_client=pb_client,
                           model_file="data/amass.urdf",
                           char_info=char_info,
                           scale=NOMINAL_H / 1.6,
                           ref_scale=NOMINAL_H / 1.6,
                           self_collision=False,
                           # actuation=spd,
                           kinematic_only=False,
                           verbose=True)
robot2 = SimAgent(name='sim_agent_1',
                            pybullet_client=pb_client,
                            model_file="data/amass.urdf",
                            char_info=char_info,
                            scale=NOMINAL_H / 1.6,
                            ref_scale=NOMINAL_H / 1.6,
                            self_collision=False,
                            # actuation=spd,
                            kinematic_only=True,
                            verbose=True)
robot2.change_visual_color([0.05, 0.05, 0.05, 0.05])

robot3 = SimAgent(name='sim_agent_2',
                            pybullet_client=pb_client,
                            model_file="data/amass.urdf",
                            char_info=char_info,
                            scale=NOMINAL_H / 1.6,
                            ref_scale=NOMINAL_H / 1.6,
                            self_collision=False,
                            # actuation=spd,
                            kinematic_only=True,
                            verbose=True)
robot3.change_visual_color([0.05, 0.05, 0.05, 0.05])

motion, imu_R, imu_acc = load_and_augment_dip_motion(robot, motion_name_gt, motion_name_imu)
viz_motion_and_imu_dip(motion, imu_R, imu_acc)
