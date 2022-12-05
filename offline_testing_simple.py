# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) Stanford University

import errno
import pickle
import importlib.util
import random
import re
import time
from datetime import datetime
from typing import Union, Tuple

import imageio
import numpy as np
from fairmotion.ops import conversions
from torch import nn

from bullet_agent import SimAgent
from real_time_runner import RTRunner
from real_time_runner_minimal import RTRunnerMin

import torch
import os
import argparse

# make deterministic
from data_utils import \
    viz_current_frame_and_store_fk_info_include_fixed, \
    loss_angle, loss_j_pos, loss_root_dist_pos, loss_max_jerk, loss_root_jerk, our_pose_2_bullet_format
from render_funcs import init_viz, COLOR_OURS, update_height_field_pb, set_color, COLOR_GT
from learning_utils import set_seed
import constants as cst

torch.set_num_threads(1)
np.set_printoptions(threshold=10_000, precision=10)
torch.set_printoptions(threshold=10_000, precision=10)


parser = argparse.ArgumentParser(description='Run our model and related works models')
parser.add_argument('--ours_path_name_kin', type=str, default="model-kin-amass-4-knee-v2.pt",
                    help='')
parser.add_argument('--name_contains', type=str, default='',
                    help='Please use "" to be able to pass multiple search keys split by whitespaces')
parser.add_argument('--test_len', type=int, default=600,
                    help='')
parser.add_argument('--render', action='store_true',
                    help='')
parser.add_argument('--compare_gt', action='store_true',
                    help='')
parser.add_argument('--seed', type=int, default=42,
                    help='')
parser.add_argument('--five_sbp', action='store_true',
                    help='')
parser.add_argument('--with_acc_sum', action='store_true',
                    help='')
parser.add_argument('--viz_terrain', action='store_true',
                    help='')
# parser.add_argument('--save_c', action='store_true',
#                     help='')                # for the DIP-IMU set which has C info
args = parser.parse_args()

set_seed(args.seed)

TEST_LEN = args.test_len
RENDER = args.render
MAX_TEST_MOTION_PRE_CAT = 50        # make testing faster
# if args.save_c:
#     MAX_TEST_MOTION_PRE_CAT = 50000
# else:
#     MAX_TEST_MOTION_PRE_CAT = 50
USE_5_SBP = args.five_sbp
WITH_ACC_SUM = args.with_acc_sum

MAP_BOUND = cst.MAP_BOUND * 2.0     # some motions are in large range
GRID_NUM = int(MAP_BOUND/cst.GRID_SIZE) * 2


def run_ours_wrapper_with_c_rt(imu, s_gt, model_name, char) -> (np.ndarray, np.ndarray):
    def load_model(name):
        from simple_transformer_with_state import TF_RNN_Past_State
        input_channels_imu = 6 * (9 + 3)
        if USE_5_SBP:
            output_channels = 18 * 6 + 3 + 20
        else:
            output_channels = 18 * 6 + 3 + 8

        model = TF_RNN_Past_State(
            input_channels_imu, output_channels,
            rnn_hid_size=512,
            tf_hid_size=1024, tf_in_dim=256,
            n_heads=16, tf_layers=4,
            dropout=0.0, in_dropout=0.0,
            past_state_dropout=0.8,
            with_acc_sum=WITH_ACC_SUM
        )
        model.load_state_dict(torch.load(name))
        model = model.cuda()
        # model.eval()
        return model

    m = load_model(model_name)

    # ours_out, c_out, viz_locs_out = test_run_ours_gpt_v4_with_c_rt(char, s_gt, imu, m, 40)
    ours_out, c_out, viz_locs_out = test_run_ours_gpt_v4_with_c_rt_minimal(char, s_gt, imu, m, 40)

    return ours_out, c_out, viz_locs_out


def test_run_ours_gpt_v4_with_c_rt_minimal(
        char: SimAgent,
        s_gt: np.array,
        imu: np.array,
        m: nn.Module,
        max_win_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # use real time runner with offline data
    rt_runner = RTRunnerMin(
        char, m, max_win_len, s_gt[0],
        with_acc_sum=WITH_ACC_SUM,
    )

    m_len = imu.shape[0]
    s_traj_pred = np.zeros((m_len, cst.n_dofs * 2))
    s_traj_pred[0] = s_gt[0]

    c_traj_pred = np.zeros((m_len, rt_runner.n_sbps * 4))
    viz_locs_seq = [np.ones((rt_runner.n_sbps, 3)) * 100.0]

    for t in range(0, m_len-1):
        res = rt_runner.step(imu[t, :], s_traj_pred[t, :3])

        s_traj_pred[t + 1, :] = res['qdq']
        c_traj_pred[t + 1, :] = res['ct']

        viz_locs = res['viz_locs']
        for sbp_i in range(viz_locs.shape[0]):
            viz_point(viz_locs[sbp_i, :], sbp_i)
        viz_locs_seq.append(viz_locs)

        if RENDER:
            time.sleep(1. / 180)

    # throw away first "trim" predictions (our algorithm gives dummy values)... append dummy value in the end.
    viz_locs_seq = np.array(viz_locs_seq)
    assert len(viz_locs_seq) == len(s_traj_pred)

    # +2 because post-processing moving average filter effectively introduce a bit more delay
    trim = rt_runner.IMU_n_smooth + 2
    s_traj_pred[0:-trim, :] = s_traj_pred[trim:, :]
    s_traj_pred[-trim:, :] = s_traj_pred[-trim-1, :]
    viz_locs_seq[0:-trim, :, :] = viz_locs_seq[trim:, :, :]
    viz_locs_seq[-trim:, :, :] = viz_locs_seq[-trim-1, :, :]

    return s_traj_pred, c_traj_pred, viz_locs_seq


def test_run_ours_gpt_v4_with_c_rt(
        char: SimAgent,
        s_gt: np.array,
        imu: np.array,
        m: nn.Module,
        max_win_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    global h_id, h_b_id

    # use real time runner with offline data
    rt_runner = RTRunner(
        char, m, max_win_len, s_gt[0],
        map_bound=MAP_BOUND,
        grid_size=cst.GRID_SIZE,
        play_back_gt=False,
        five_sbp=USE_5_SBP,
        with_acc_sum=WITH_ACC_SUM,
        multi_sbp_terrain_and_correction=False
    )

    m_len = imu.shape[0]
    s_traj_pred = np.zeros((m_len, cst.n_dofs * 2))
    c_traj_pred = np.zeros((m_len, rt_runner.n_sbps * 4))
    s_traj_pred[0] = s_gt[0]

    viz_locs_seq = [np.ones((rt_runner.n_sbps, 3)) * 100.0]

    for t in range(0, m_len-1):
        res = rt_runner.step(imu[t, :], s_traj_pred[t, :3], t=t)

        s_traj_pred[t + 1, :] = res['qdq']
        c_traj_pred[t + 1, :] = res['ct']

        viz_locs = res['viz_locs']
        for sbp_i in range(viz_locs.shape[0]):
            viz_point(viz_locs[sbp_i, :], sbp_i)

        viz_locs_seq.append(viz_locs)

        if t % 15 == 0 and h_id is not None:
            # TODO: double for loop...
            for ii in range(init_grid_np.shape[0]):
                for jj in range(init_grid_np.shape[1]):
                    init_grid_list[jj*init_grid_np.shape[0]+ii] = \
                        rt_runner.region_height_list[rt_runner.height_region_map[ii, jj]]
            h_id, h_b_id = update_height_field_pb(
                pb_client,
                h_data=init_grid_list,
                scale=cst.GRID_SIZE,
                terrainShape=h_id,
                terrain=h_b_id
            )

        if RENDER:
            time.sleep(1. / 180)

    # throw away first "trim" predictions (our algorithm gives dummy values)... append dummy value in the end.
    viz_locs_seq = np.array(viz_locs_seq)

    # +2 because post-processing moving average filter effectively introduce a bit more delay
    trim = rt_runner.IMU_n_smooth + 2
    s_traj_pred[0:-trim, :] = s_traj_pred[trim:, :]
    s_traj_pred[-trim:, :] = s_traj_pred[-trim-1, :]
    viz_locs_seq[0:-trim, :, :] = viz_locs_seq[trim:, :, :]
    viz_locs_seq[-trim:, :, :] = viz_locs_seq[-trim-1, :, :]

    return s_traj_pred, c_traj_pred, viz_locs_seq


def viz_2_trajs_and_return_fk_records_with_sbp(
        char1: SimAgent,
        char2: SimAgent,
        traj1: np.ndarray,
        traj2: np.ndarray,
        start_t: int,
        end_t: int,
        gui: bool,
        seq_c_viz: Union[np.ndarray, None],
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):

    m_len = len(traj1)      # use first length if mismatch

    pq_g_1_s = []
    pq_g_2_s = []

    for t in range(start_t, m_len-end_t):

        pq_g_2 = viz_current_frame_and_store_fk_info_include_fixed(char2, traj2[t])
        pq_g_1 = viz_current_frame_and_store_fk_info_include_fixed(char1, traj1[t])   # GT in grey

        pq_g_1_s.append(pq_g_1)
        pq_g_2_s.append(pq_g_2)

        if seq_c_viz is not None:
            cur_c_viz = seq_c_viz[t, :, :]
            for sbp_i in range(cur_c_viz.shape[0]):
                viz_point(cur_c_viz[sbp_i, :], sbp_i)

        if gui:
            time.sleep(1. / 180)

    return traj1[start_t: m_len-end_t], traj2[start_t: m_len-end_t], np.array(pq_g_1_s), np.array(pq_g_2_s)


def post_processing_our_model(
        char: SimAgent,
        ours_out: np.ndarray) -> np.ndarray:
    poses_post = []
    for pose in ours_out:
        pose_post = our_pose_2_bullet_format(char, pose)
        poses_post.append(pose_post.tolist())
    poses_post = np.array(poses_post)

    return poses_post


def viz_point(x, ind):
    pb_client.resetBasePositionAndOrientation(
        VIDs[ind],
        x,
        [0., 0, 0, 1]
    )


def get_all_testing_filenames(name_contains_list):
    file_paths = []
    for src_dir in imu_readings_dirs_OUR_format:
        src_dir = os.path.join("data", src_dir)
        # list_dirs = [x[0] for x in os.walk(src_dir)]
        # for d in list_dirs:
        with os.scandir(src_dir) as it:
            for entry in it:
                n = entry.name
                if n.endswith('pkl'):
                    f_path = os.path.join(src_dir, n)
                    for name_contains in name_contains_list:
                        if re.search(name_contains, f_path, re.IGNORECASE):
                            file_paths.append(f_path)

                            break
                        # break to here
    return file_paths


"""
    main
"""

imu_readings_dirs_OUR_format = [
    "syn_AMASS_CMU_v0", "syn_Eyes_Japan_Dataset_v0",
    "syn_KIT_v0", "syn_HUMAN4D_v0",
    "syn_ACCAD_v0", "syn_DFaust_67_v0", "syn_HumanEva_v0", "syn_MPI_Limits_v0",
    "syn_MPI_mosh_v0", "syn_SFU_v0", "syn_Transitions_mocap_v0",
    "preprocessed_DIP_IMU_v0", "preprocessed_TotalCapture_v0", "syn_TotalCapture_v0",
    "syn_DanceDB_v0"
]

# if args.save_c:
#     try:
#         os.makedirs("../release/data/preprocessed_DIP_IMU_v0_c")    # store c here
#     except FileExistsError:
#         print("warning: path existed")
#     except OSError:
#         exit()

''' Load Character Info Moudle '''
spec = importlib.util.spec_from_file_location(
    "char_info", "amass_char_info.py")
char_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(char_info)

name_contains_l = args.name_contains.split()
print(name_contains_l)
test_files = get_all_testing_filenames(name_contains_l)
print(len(test_files))
if len(test_files) > MAX_TEST_MOTION_PRE_CAT:
    test_files = random.sample(test_files, MAX_TEST_MOTION_PRE_CAT)
print(test_files)

color = COLOR_OURS


# TODO: really odd, need to be huge for pybullet to work (say. 10.0)
init_grid_np = np.random.uniform(-10.0, 10.0, (GRID_NUM, GRID_NUM))
init_grid_list = list(init_grid_np.flatten())

pb_client, c1, c2, VIDs, h_id, h_b_id = init_viz(char_info,
                                                 init_grid_list,
                                                 hmap_scale=cst.GRID_SIZE,
                                                 gui=RENDER,
                                                 compare_gt=args.compare_gt,
                                                 color=color,
                                                 viz_h_map=args.viz_terrain)

gt_list = []
ours_list = []
ours_c_list = []
ours_c_viz_list = []
tp_list = []
dip_list = []
test_files_included = []
for f in test_files:

    if not (os.path.exists(f)):
        print("ignored ", f)
        continue

    data = pickle.load(open(f, "rb"))
    X = data['imu']
    Y = data['nimble_qdq']

    # exclude too short trajs
    if Y.shape[0] < 2.5 / cst.DT:
        continue

    # to make all motion equal in stat compute, and run faster
    if Y.shape[0] > TEST_LEN:
        rand_start = random.randrange(0, Y.shape[0] - TEST_LEN)
        start = rand_start
        end = rand_start + TEST_LEN
    else:
        start = 0
        end = Y.shape[0]
    X = X[start: end, :]
    Y = Y[start: end, :]

    # for clearer visualization, amass data not calibrated well wrt floor
    # translation errors are computed from displacement not absolute Y
    Y[:, 2] += 0.05       # move motion root 5 cm up

    # print(X.shape)
    # print(Y.shape)
    # print(start)
    # print(end)
    gt_list.append(Y)
    test_files_included.append(f)
    print(f)

    ours, C, ours_c_viz = run_ours_wrapper_with_c_rt(X, Y, args.ours_path_name_kin, c1)
    ours_list.append(ours)
    ours_c_viz_list.append(ours_c_viz)

    # if args.save_c:
    #     save_name = f.replace("v0", "v0_c")
    #     assert "dipimu" in f
    #     with open(save_name, "wb") as handle:
    #         assert len(X) == len(C)
    #         print(C.shape)
    #         pickle.dump(
    #             {"constrs": C},
    #             handle,
    #             protocol=pickle.HIGHEST_PROTOCOL
    #         )
    #     print("saved", save_name)

if args.compare_gt:
    with open("test-output-tmp.pkl", "wb") as handle:
        pickle.dump({"gt_list": gt_list,
                     "ours_list": ours_list,
                     "tp_list": tp_list,
                     "dip_list": dip_list},
                    handle, protocol=pickle.HIGHEST_PROTOCOL)

    losses_angle = []
    losses_j_pos = []
    losses_2s_root = []
    losses_5s_root = []
    losses_10s_root = []
    losses_jerk_max = []
    losses_jerk_root = []
    for i in range(len(gt_list)):
        traj_1 = post_processing_our_model(c1, gt_list[i])

        traj_2 = post_processing_our_model(c1, ours_list[i])

        ours_c_viz = ours_c_viz_list[i] if len(ours_c_viz_list) > 0 else None

        res_tuple = viz_2_trajs_and_return_fk_records_with_sbp(
            c2, c1, traj_1, traj_2, 30, 6, RENDER, ours_c_viz)      # first 0.5s uninteresting

        losses_angle.append(loss_angle(*res_tuple))
        losses_j_pos.append(loss_j_pos(*res_tuple))
        losses_2s_root.append(loss_root_dist_pos(*res_tuple, t=2.0))
        losses_5s_root.append(loss_root_dist_pos(*res_tuple, t=5.0))
        losses_10s_root.append(loss_root_dist_pos(*res_tuple, t=10.0))
        losses_jerk_max.append(loss_max_jerk(*res_tuple))
        losses_jerk_root.append(loss_root_jerk(*res_tuple))

    print(np.mean(losses_angle))
    print(np.mean(losses_j_pos))
    print(np.mean(losses_2s_root))
    print(np.mean(losses_5s_root))
    print(np.mean(losses_10s_root))
    print(np.mean(losses_jerk_max))
    print(np.mean(losses_jerk_root))

    print(np.max(losses_angle), test_files_included[np.argmax(losses_angle)])
    print(np.max(losses_j_pos), test_files_included[np.argmax(losses_j_pos)])
    print(np.max(losses_2s_root), test_files_included[np.argmax(losses_2s_root)])
    print(np.max(losses_5s_root), test_files_included[np.argmax(losses_5s_root)])
    print(np.max(losses_10s_root), test_files_included[np.argmax(losses_10s_root)])
    print(np.max(losses_jerk_max), test_files_included[np.argmax(losses_jerk_max)])
    print(np.max(losses_jerk_root), test_files_included[np.argmax(losses_jerk_root)])
