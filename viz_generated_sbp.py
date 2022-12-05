# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) Stanford University
# untested helper script for your convenience

import errno
import pickle
import importlib.util
import random
import re
import time
from datetime import datetime

import imageio
import numpy as np
import pybullet as pb
from fairmotion.ops import conversions
from fairmotion.ops import quaternion
from scipy import ndimage

import bullet_client
import bullet_utils as bu
import sim_agent

import os
import subprocess
import argparse

# make deterministic
from data_utils import get_rot_center_sample_based, viz_current_frame_and_store_fk_info_include_fixed, \
    our_pose_2_bullet_format
from transformers.utils import set_seed
set_seed(12345)
np.set_printoptions(threshold=10_000, precision=6)


FRAME_DELTA = 2
DT = 1./60
n_DoF = 57
NOMINAL_H = 1.7


def post_processing_our_model(char: sim_agent.SimAgent, ours_out: np.ndarray) -> np.ndarray:
    poses_post = []
    for pose in ours_out:
        pose_post = our_pose_2_bullet_format(char, pose)
        poses_post.append(pose_post.tolist())
    poses_post = np.array(poses_post)

    return poses_post


def viz_traj_and_return_records(
        char: sim_agent.SimAgent,
        char2: sim_agent.SimAgent,
        traj: np.ndarray,
        traj2: np.ndarray,
        start_t: int):

    m_len = traj.shape[0]
    # TP & DIP do not predict final frames
    end_t = m_len

    pq_g_s = []

    link = char.get_char_info().lankle
    # link = char.get_char_info().lwrist
    # link = char.get_char_info().root

    #
    p1_l = []
    # p2_l = []

    r_prev = None
    residue_sum = np.array([0., 0, 0])

    for t in range(start_t, end_t):

        traj2[t, :3] -= residue_sum

        _ = viz_current_frame_and_store_fk_info_include_fixed(char2, traj2[t])

        pq_g = viz_current_frame_and_store_fk_info_include_fixed(char, traj[t])
        pq_g_s.append(pq_g)

        if t > start_t + FRAME_DELTA:

            cur_pq = pq_g_s[t-start_t][link + 1]
            cur_p, cur_q = cur_pq[:3], cur_pq[3:]

            prev_pq = pq_g_s[t-start_t - FRAME_DELTA][link + 1]
            prev_p, prev_q = prev_pq[:3], prev_pq[3:]

            # r, w = get_rot_center(prev_p, prev_q, cur_p, cur_q, FRAME_DELTA * DT, r_prev)
            r, residue = get_rot_center_sample_based(prev_p, prev_q, cur_p, cur_q, FRAME_DELTA * DT, r_prev, link)

            residue_sum += residue * DT

            # cur_R = conversions.Q2R(cur_q)
            # pb_c.addUserDebugLine(list(cur_p), list(cur_p + cur_R[:, 0]), (1, 0, 0), 3)
            # pb_c.addUserDebugLine(list(cur_p), list(cur_p + cur_R[:, 1]), (0, 1, 0), 3)
            # pb_c.addUserDebugLine(list(cur_p), list(cur_p + cur_R[:, 2]), (0, 0, 1), 3)

            if r is None:
                r = np.array([100., 100., 100.])
                r_prev = None
            else:
                r_prev = r

            viz_point((cur_p + prev_p) / 2.0 + r, 0)  # +1 for root info
            # viz_point(cur_p, 1)  # +1 for root info
            p1_l.append(residue_sum.copy())
            # p2_l.append(p2)

            # input("press enter")

        if RENDER:
            time.sleep(1. / 120.0)

            # H = 1280
            # W = 720
            # img = pb_c.getCameraImage(
            #     H, W,
            #     renderer=pb_c.ER_BULLET_HARDWARE_OPENGL)
            # rgb = np.reshape(img[2], (W, H, 4))
            # output_rgb = os.path.join(dir_name, "s%05d.png" % t)
            # imageio.imwrite(output_rgb, rgb[:, :, :3])  # alpha channel dropped

        pb_c.removeAllUserDebugItems()

    print(residue_sum)
    p1_l = np.array(p1_l)
    from matplotlib import pyplot as plt
    plt.subplot(3, 1, 1)
    plt.plot(p1_l[:, 0])
    plt.plot(p1_l[:, 0])

    plt.subplot(3, 1, 2)
    plt.plot(p1_l[:, 1])
    plt.plot(p1_l[:, 1])

    plt.subplot(3, 1, 3)
    plt.plot(p1_l[:, 2])
    plt.plot(p1_l[:, 2])
    plt.show()

    return traj[start_t: end_t], np.array(pq_g_s),


# if v has a large component along w, predict 0;
# else, trim the w component in v, so w x r = v' must have many solutions
# run least square, find the minimal r solution

# def get_rot_center(x1, q1, x2, q2):
#     ps_1 = np.tile(x1[:, np.newaxis], (1, 3)) + conversions.Q2R(q1) * 100.0
#     ps_2 = np.tile(x2[:, np.newaxis], (1, 3)) + conversions.Q2R(q2) * 100.0
#
#     b = ps_1[0, :] ** 2 + ps_1[1, :] ** 2 + ps_1[2, :] ** 2 - \
#         ps_2[0, :] ** 2 + ps_2[1, :] ** 2 + ps_2[2, :] ** 2
#
#     A = 2 * (ps_1 - ps_2)
#
#     rc = np.linalg.solve(A.T, b)
#     print(rc)
#     return rc


def init_viz(_char_info):
    m = pb.GUI if RENDER else pb.DIRECT
    pb_c = bullet_client.BulletClient(
        connection_mode=m)
    pb_c.resetSimulation()

    # _ = \
    #     pb_c.loadURDF(
    #         "plane_implicit.urdf",
    #         [0, 0, 0],
    #         [0, 0, 0, 1.0],
    #         useMaximalCoordinates=True)

    pb_c.configureDebugVisualizer(
        flag=pb.COV_ENABLE_SHADOWS | \
             pb.COV_ENABLE_RENDERING | \
             pb.COV_ENABLE_WIREFRAME,
        enable=1,
        shadowMapResolution=2048,
        shadowMapIntensity=0.3,
        shadowMapWorldSize=10,
        rgbBackground=[1, 1, 1],
        lightPosition=(5.0, 5.0, 10.0))

    r = sim_agent.SimAgent(name='sim_agent_1',
                           pybullet_client=pb_c,
                           model_file="data/amass.urdf",
                           char_info=_char_info,
                           scale=NOMINAL_H / 1.6,
                           ref_scale=NOMINAL_H / 1.6,
                           self_collision=False,
                           # actuation=spd,
                           kinematic_only=True,
                           verbose=True)
    r.change_visual_color([0.5, 0.5, 0.5, 0.5])

    r2 = sim_agent.SimAgent(name='sim_agent_2',
                            pybullet_client=pb_c,
                            model_file="data/amass.urdf",
                            char_info=_char_info,
                            scale=NOMINAL_H / 1.6,
                            ref_scale=NOMINAL_H / 1.6,
                            self_collision=False,
                            # actuation=spd,
                            kinematic_only=True,
                            verbose=True)
    r2.change_visual_color([1.0, 0.8, 0.0, 1.0])

    # dt_sim = 1. / 480  # control freq 30Hz
    # pb_c.setTimeStep(dt_sim)
    # pb_c.setPhysicsEngineParameter(numSubSteps=2)
    # pb_c.setPhysicsEngineParameter(numSolverIterations=10)

    p_vids = []
    for i in range(10):
        color = [1.0, 0.2, 0.2, 1.0]
        visual_id = pb_c.createVisualShape(pb_c.GEOM_SPHERE,
                                           radius=0.05,
                                           rgbaColor=color,
                                           specularColor=[1, 1, 1])
        bid = pb_c.createMultiBody(0.0,
                                   -1,
                                   visual_id,
                                   [100.0, 100.0, 100.0],
                                   [0.0, 0.0, 0.0, 1.0])
        p_vids.append(bid)
        # self._p.setCollisionFilterGroupMask(bid, -1, 0, 0)

    return pb_c, r, r2, p_vids


def viz_point(x, ind):
    pb_c.resetBasePositionAndOrientation(
        vids[ind],
        x,
        [0., 0, 0, 1]
    )


parser = argparse.ArgumentParser(description='visualize gt motion')
parser.add_argument('--motion_name', type=str, default='',
                    help='')
parser.add_argument('--start_t', type=int, default=10,
                    help='')
parser.add_argument('--render', action='store_true',
                    help='')
args = parser.parse_args()
RENDER = args.render

''' Load Character Info Moudle '''
spec = importlib.util.spec_from_file_location(
    "char_info", "amass_char_info.py")
char_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(char_info)
pb_c, c, c2, vids = init_viz(char_info)

# dir_name = 'img' + datetime.now().strftime('%T').replace(':', '-')
# if not os.path.exists(dir_name):
#     print("does not exist")
#     try:
#         os.makedirs(dir_name)
#     except OSError as exc:  # Guard against race condition
#         if exc.errno != errno.EEXIST:
#             raise

f = "syn_SFU_v0/0017_0017_JumpAndRoll001.pkl"
data = pickle.load(open(f, "rb"))
X = data['imu']
Y = data['nimble_qdq']

# Y[:, :3] = ndimage.uniform_filter1d(
#         Y[:, :3], 11, axis=0, mode="nearest"
#     )
# Y[:, 6:] = ndimage.uniform_filter1d(
#         Y[:, 6:], 11, axis=0, mode="nearest"
#     )

Y_post = post_processing_our_model(c, Y)
Y_post_copy = Y_post.copy()
_ = viz_traj_and_return_records(c, c2, Y_post, Y_post_copy, args.start_t)
