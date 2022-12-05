# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) Stanford University

import argparse
import importlib.util
import os
import pickle
import re
import sys
from typing import List, Dict
import random
from joblib import Parallel, delayed

import numpy as np
import pybullet as pb
import pybullet_data
from fairmotion.core.motion import Motion
from fairmotion.data import amass
from fairmotion.ops import conversions

from data_utils import get_rot_center_sample_based, \
    get_raw_motion_info_nimble_q_dummy_dq
import bullet_client
from bullet_agent import SimAgent
import constants as cst
from learning_utils import set_seed

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

# q dq in joint order of NimblePhysics, though dq is not used except for root
USE_KNEE_RATHER_ANKLE_IMU = True
print("knee imu?", USE_KNEE_RATHER_ANKLE_IMU)

set_seed(42)


def load_motion_amass(motion_file):
    m = amass.load(file=motion_file, bm_path="data/smplh/neutral/model.npz", model_type="smplh")
    # m = MotionWithVelocity.from_motion(m)     # only need q, no dq
    return m


def get_raw_motion_info(robot: SimAgent, m: Motion) -> List[Dict[int, List[float]]]:
    # runs robot FK
    raw_info = []
    cur_time = 0.015 / 2.0

    while cur_time < m.length():
        cur_pose = m.get_pose_by_time(cur_time)
        robot.set_pose(cur_pose, None)

        cur_info = {}

        all_joint_idx = robot.get_char_info().joint_idx.values()

        for idx in all_joint_idx:
            if idx == robot.get_char_info().root:
                # this is the root joint
                _, Q, _, _ = robot.get_root_state()
                p = robot.get_root_local_point_p(cst.ROOT_COM_OFFSET)
                cur_info[idx] = list(p) + list(Q)
            else:
                state = robot.get_link_states([idx])
                cur_info[idx] = list(state[0]) + list(state[1])
            assert len(cur_info[idx]) == 7

        # print(cur_info)
        # input("press enter")
        raw_info.append(cur_info)
        cur_time += cst.DT

    return raw_info


def get_all_contr_seqs_from_raw_motion_info(
    robot: SimAgent,
    raw_info: List[Dict[int, List[float]]]
) -> np.ndarray:

    _info = robot.get_char_info()
    constr_links = [
            _info.lankle,
            _info.rankle,
            _info.lwrist,
            _info.rwrist,
            _info.root,
    ]

    c_all = []
    for link in constr_links:
        contrs_link, r_sum_link = get_link_contr_seq_from_raw_motion_info(
            raw_info, link
        )
        c_all.append(contrs_link)
        # if np.max(np.abs(r_sum_link)) > 0.2:
        #     print(link, r_sum_link)

    c_all = np.concatenate(c_all, axis=1)

    return c_all


def get_link_contr_seq_from_raw_motion_info(
        raw_info: List[Dict[int, List[float]]],
        link: int
) -> (np.ndarray, np.ndarray):
    # a l-by-4 matrix, each row being
    # (0/1, (0,0,0)/Rr)

    r_prev = None
    m_len = len(raw_info)
    contr_seq = np.zeros((m_len, 4))

    residue_sum = np.array([0., 0, 0])

    def get_p_q_from_t(_t):
        cur_info = raw_info[_t]
        link_p = np.array(cur_info[link][:3])
        link_q = np.array(cur_info[link][3:])
        return link_p, link_q

    for t in range(2, m_len - 2):

        prev_p, prev_q = get_p_q_from_t(t - 1)
        cur_p, cur_q = get_p_q_from_t(t)
        next_p, next_q = get_p_q_from_t(t + 1)

        r, residue = get_rot_center_sample_based(prev_p, prev_q, next_p, next_q, 2 * cst.DT, r_prev, link)

        residue_sum += residue * cst.DT

        # if r is None, zeros filled in already
        if r is not None:
            cur_R = conversions.Q2R(cur_q)
            assert np.linalg.norm(r) < 0.25
            contr_seq[t, :] = np.concatenate((
                [1.0],
                r
            ))

        r_prev = r

    return contr_seq, residue_sum


def get_imu_readings_from_raw_motion_info(
        robot: SimAgent,
        raw_info: List[Dict[int, List[float]]]
) -> np.ndarray:
    # a matrix of size l-by-(6*(9+3))

    l = len(raw_info)
    H = np.zeros((l, 6 * (9 + 3)), float)
    _info = robot.get_char_info()

    if USE_KNEE_RATHER_ANKLE_IMU:
        imu_joints = [
            _info.root,
            _info.lwrist,
            _info.rwrist,
            _info.lknee,
            _info.rknee,
            _info.upperneck,
        ]
    else:
        imu_joints = [
            _info.root,
            _info.rankle,
            _info.lankle,
            _info.lwrist,
            _info.rwrist,
            _info.upperneck
        ]

    for t in range(l):
        # fill in the ori readings
        imu_Rs = []  # 6*9
        cur_info = raw_info[t]

        root_q = np.array(cur_info[_info.root][3:])
        root_R = conversions.Q2R(root_q)
        imu_Rs += root_R.flatten().tolist()  # in global frame (no transform)

        for j in imu_joints[1:]:
            # other five IMUs
            joint_R = conversions.Q2R(np.array(cur_info[j][3:]))
            imu_Rs += joint_R.flatten().tolist()

        H[t, :6 * 9] = imu_Rs

    for t in range(cst.acc_fd_N, l - cst.acc_fd_N):
        # fill in the acc readings
        imu_as = []
        cur_info = raw_info[t]
        prev_info = raw_info[t - cst.acc_fd_N]
        next_info = raw_info[t + cst.acc_fd_N]

        root_a = \
            -2 * np.array(cur_info[_info.root][:3]) \
            + next_info[_info.root][:3] \
            + prev_info[_info.root][:3]
        root_a /= (cst.DT_FIN_ACC ** 2)
        imu_as += root_a.tolist()

        for j in imu_joints[1:]:
            # other five IMUs
            joint_a = -2 * np.array(cur_info[j][:3]) + next_info[j][:3] + prev_info[j][:3]
            joint_a /= (cst.DT_FIN_ACC ** 2)
            imu_as += joint_a.flatten().tolist()

        H[t, 6 * 9:] = imu_as

    # pad boundary acc
    H[:cst.acc_fd_N, 6 * 9:] = H[cst.acc_fd_N, 6 * 9:]
    H[-cst.acc_fd_N:, 6 * 9:] = H[-cst.acc_fd_N - 1, 6 * 9:]

    return H


def gen_data_job_single_core(src_dir_name, save_dir, file_name, name_contains):

    if not file_name.endswith('_poses.npz'):
        return 0

    try:
        # use pybullet for forward kinematics
        # p.GUI if want to visualize data gen (num_core = 1)
        Mode = pb.DIRECT
        pb_client = bullet_client.BulletClient(
            connection_mode=Mode)
        pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb_client.resetSimulation()

        motion_name = os.path.join(src_dir_name, file_name)

        ext = ".pkl"
        save_name_local = src_dir_name.rsplit('/', 1)[-1] + "_" + file_name[:-10] + ext
        save_name = save_dir + "/" + save_name_local
        save_name = save_name.replace(" ", "_")

        if len(name_contains) > 0 and not re.search(name_contains, save_name, re.IGNORECASE):
            print(save_name, " not matching specified subset, ignore")
            return 0
        if os.path.exists(save_name):
            print(save_name, " already generated")
            return 0

        h = cst.NOMINAL_H * random.uniform(0.9, 1.1)
        # h = cst.NOMINAL_H

        robot = SimAgent(name='sim_agent_0',
                         pybullet_client=pb_client,
                         model_file="data/amass.urdf",
                         char_info=char_info,
                         scale=h / 1.6,
                         ref_scale=h / 1.6,
                         self_collision=False,
                         # actuation=spd,
                         kinematic_only=True,
                         verbose=True)

        motion = load_motion_amass(motion_name)
        motion_info = get_raw_motion_info(robot, motion)

        imu = get_imu_readings_from_raw_motion_info(robot, motion_info)

        contrs = get_all_contr_seqs_from_raw_motion_info(robot, motion_info)

        qdq = get_raw_motion_info_nimble_q_dummy_dq(robot, motion)

        print(motion_name, '\n', save_name, '\n', h, imu.shape, contrs.shape, qdq.shape, '\n')
        with open(save_name, "wb") as handle:
            pickle.dump(
                {"imu": imu, "nimble_qdq": qdq, "constrs": contrs},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        # pb_client.resetSimulation()
        return 1
    except Exception as e:
        print("ignored: ", file_name, "error: ", e)
        return 0


def gen_data_all(save_dir, src_dir, name_contains):
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        print("warning: path existed")
    except OSError:
        exit()

    list_dirs = [x[0] for x in os.walk(src_dir)]

    count = 0
    for d in list_dirs:
        with os.scandir(d) as it:

            # https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop
            # results = Parallel(n_jobs=2)(delayed(countdown)(10 ** 7) for _ in range(20))
            # with parallel_backend("loky", inner_max_num_threads=2):

            results = Parallel(n_jobs=args.n_proc)(
                delayed(gen_data_job_single_core)(
                    d, save_dir, entry.name, name_contains
                ) for entry in it
            )
            count += np.sum(results)

    print("count ", count)


parser = argparse.ArgumentParser(description='Generate Synthetic IMU Data from AMASS')
parser.add_argument('--train_eval_split', type=float, default=1.0,
                    help='store subset in sub folder called eval')
parser.add_argument('--save_dir', type=str, default='tmp',
                    help='')
parser.add_argument('--src_dir', type=str, default='none',
                    help='load all from srouce dir rather than from motion list file')
parser.add_argument('--motion_list_file', type=str, default='none.txt',
                    help='')
parser.add_argument('--name_contains', type=str, default='',
                    help='')
parser.add_argument('--n_proc', type=int, default=7,
                    help='')
args = parser.parse_args()

''' Load Character Info Moudle '''
spec = importlib.util.spec_from_file_location(
    "char_info", "amass_char_info.py")
char_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(char_info)

v_up = char_info.v_up_env
assert np.allclose(np.linalg.norm(v_up), 1.0)

gen_data_all(args.save_dir, args.src_dir, args.name_contains)
