# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) Stanford University

import re

import numpy as np
import time
import os
import pickle

from scipy import ndimage
from data_utils import batch_to_rot_mat_2axis, imu_rotate_to_local
import constants as cst


def store_imu_s_info(imu_gt_dirs,
                     down_sample_rates,
                     imu_save_name,
                     s_save_name,
                     info_save_name,
                     num_sbps=5,
                     ):
    info = []  # each row: start frame, end frame, down sample rate
    IMU = []    # width 6 * 9 (IMU rotations) + 6 * 3 (IMU accelerations)
    IMU_acc_sums = []   # width 6 * 3 (IMU accelerations)
    S_2axis = []    # width 54 (18 joints, including root rot) / 3 (originally axis-angles) * 6 (in two axis representation) + 3 (root vel)
    C = []      # 5 (SBPs) * 4
    count = 0
    start_f = 0
    end_f = 0

    start_time = time.time()

    assert len(imu_gt_dirs) == len(down_sample_rates)
    for imu_gt_dir, down_sample_rate in zip(imu_gt_dirs, down_sample_rates):

        # the DIP dataset does not have root translation info
        # so root vel info does not make sense as well
        # set corresponding entries to nan, so they can be excluded from loss calc

        is_augmented_dip = ("preprocessed_DIP_IMU" in imu_gt_dir)

        files = []
        for f in sorted(os.listdir(imu_gt_dir)):
            f_full = imu_gt_dir + "/" + f
            if f_full.endswith(".pkl") and os.path.isfile(f_full):
                if len(name_contains_l) > 0:
                    for name_contains in name_contains_l:
                        if re.search(name_contains, f_full, re.IGNORECASE):
                            files.append(f_full)
                            print(f_full)
                            break
                else:
                    files.append(f_full)

        for file in files:

            IMU_GT_cur = pickle.load(open(file, "rb"))
            IMU_cur = IMU_GT_cur["imu"]
            S_cur = IMU_GT_cur["nimble_qdq"]
            if is_augmented_dip:
                S_cur[:, cst.n_dofs:(cst.n_dofs + 3)] = np.nan
            C_cur = IMU_GT_cur["constrs"]

            # some minor mismatch between qdq & imu readings length for real DIP data
            assert np.abs(len(IMU_cur) - len(S_cur)) <= 1
            m_len = min(S_cur.shape[0], IMU_cur.shape[0])
            if m_len <= cst.ACC_SUM_WIN_LEN:
                print("too short: ", file)
                continue
            else:
                count += 1

            IMU_cur = IMU_cur[4:m_len - 4]
            S_cur = S_cur[4:m_len - 4]
            C_cur = C_cur[4:m_len - 4]
            end_f += m_len - 8

            # acc average filter
            # note the mode is 'nearest', padding first reading to the left
            # test time will match this setting.
            IMU_cur[:, 6 * 9: 6 * 9 + 18] = ndimage.uniform_filter1d(
                IMU_cur[:, 6 * 9: 6 * 9 + 18], cst.ACC_MOVING_AVE_LEN, axis=0, mode="nearest"
            )
            # constant bias noise
            IMU_cur[:, 6 * 9: 6 * 9 + 18] += np.random.uniform(-cst.BIAS_NOISE_ACC, cst.BIAS_NOISE_ACC, 18)
            IMU_local = imu_rotate_to_local(IMU_cur)
            IMU.append(np.single(IMU_local))

            # padding acc sum features
            IMU_local_acc = IMU_local[:, 6 * 9: 6 * 9 + 18]
            b = np.cumsum(IMU_local_acc, axis=0)
            b[cst.ACC_SUM_WIN_LEN:, :] = b[cst.ACC_SUM_WIN_LEN:, :] - b[:-cst.ACC_SUM_WIN_LEN, :]
            IMU_local_acc_sum = b / cst.ACC_SUM_DOWN_SCALE
            IMU_acc_sums.append(np.single(IMU_local_acc_sum))

            S_cur_2axis = np.single(batch_to_rot_mat_2axis(S_cur[:, 3:(cst.n_dofs + 3)]))
            S_2axis.append(S_cur_2axis)

            assert num_sbps * 4 == C_cur.shape[1]
            C_cur = np.single(C_cur)
            C.append(C_cur)

            info.append([start_f, end_f, down_sample_rate])
            print([start_f, end_f, down_sample_rate])

            start_f = end_f

    print("time here", time.time() - start_time)

    IMU = np.concatenate(IMU, axis=0)
    np.save(imu_save_name, IMU)
    print("IMU shape", IMU.shape)
    del IMU

    IMU_acc_sums = np.concatenate(IMU_acc_sums, axis=0)
    np.save(imu_save_name.replace("imu", "sum_imu"), IMU_acc_sums)
    print("IMU acc sum shape", IMU_acc_sums.shape)
    del IMU_acc_sums

    info = np.array(info)
    np.save(info_save_name, np.array(info))
    print("info shape", info.shape)
    del info

    S_2axis = np.concatenate(S_2axis, axis=0)
    C = np.concatenate(C, axis=0)
    S_all = np.concatenate((S_2axis, C), axis=1)
    del S_2axis, C
    np.save(s_save_name, S_all)
    print("S shape", S_all.shape)
    del S_all

    print("num motions", count)
    print("load time", time.time() - start_time)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Preprocess and merge all syn and real training data')
    parser.add_argument('--data_version_tag', type=str, default="v1", help='')
    args = parser.parse_args()
    TAG = args.data_version_tag

    name_contains_l = []  # default, train on everything
    names_all = ""

    # # what subset among AMASS data to get
    # name_contains_l = ["stair"]
    # names_all = ""
    # for name in name_contains_l:
    #     names_all += '_'
    #     names_all += name
    #
    # print(names_all)

    dataset_names = ["data/syn_AMASS_CMU_v0", "data/syn_Eyes_Japan_Dataset_v0",
                     "data/syn_KIT_v0", "data/syn_HUMAN4D_v0",
                     "data/syn_ACCAD_v0", "data/syn_DFaust_67_v0", "data/syn_HumanEva_v0", "data/syn_MPI_Limits_v0",
                     "data/syn_MPI_mosh_v0", "data/syn_SFU_v0", "data/syn_Transitions_mocap_v0",
                     "data/syn_TotalCapture_v0", "data/preprocessed_DIP_IMU_v0_with_aug_c_train"]

    for dataset_name in dataset_names:
        dataset_name.replace("v0", TAG)

    # downweight larger datasets in AMASS a bit -- probably unimportant
    # note that only syn_TotalCapture is used in training, not the real preprocessed_TotalCapture
    dataset_down_sample_rates = [100, 100, 250, 100, 60, 60, 60, 60, 60, 60, 60, 60, 60]

    store_imu_s_info(
        imu_gt_dirs=dataset_names,
        down_sample_rates=dataset_down_sample_rates,
        imu_save_name="data/imu_train_" + TAG + names_all,
        s_save_name="data/s_train_" + TAG + names_all,
        info_save_name='data/info_train_' + TAG + names_all,
        num_sbps=5,
    )


