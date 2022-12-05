# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) Stanford University

from typing import Dict, Union, Tuple

import numpy as np
import torch
from fairmotion.ops import conversions

from bullet_agent import SimAgent

from data_utils import \
    batch_to_rot_mat_2axis, batch_rot_mat_2axis_to_aa, our_pose_2_bullet_format, \
    viz_current_frame_and_store_fk_info_include_fixed, get_cur_step_root_correction_from_all_constr, \
    imu_rotate_to_local
import constants as cst


class RTRunnerMin:
    def __init__(
        self,
        char: SimAgent,
        model_kin: torch.nn.Module,
        max_input_l: int,
        s_init: np.array,
        with_acc_sum=True,
    ):
        self.n_sbps = 5     # minimal version, only use 2 SBPs on feet though predict 5
        self.with_acc_sum = with_acc_sum

        self.model = model_kin
        self.char = char

        self.s_and_c_in_buffer = []     # (18*6 + 3 + n_sbps * 4)
        self.raw_imu_buffer = []        # (72)
        self.smoothed_imu_buffer = []   # (72)
        self.imu_acc_sum_buffer = []    # (18)
        self.s_c_smooth_buffer = []     # (18*6 + 3 + n_sbps * 4)
        self.pq_g_buffer = []           # history list of global pos&ori of all bodies

        self.c_locs = np.ones((self.n_sbps, 3)) * 100.0

        self.s_init = s_init            # (2 * N_DOFS, ) (q, dq) dq not predicted except root vel though
        self.last_s = None              # (2 * N_DOFS, ) (q, dq) dq not predicted except root vel though
        self.record_state_aa_and_c(s_init, np.zeros(self.n_sbps * 4))

        s_init_bullet = our_pose_2_bullet_format(self.char, s_init)
        pq_g = viz_current_frame_and_store_fk_info_include_fixed(
            self.char, s_init_bullet)
        self.pq_g_buffer.append(pq_g)

        self.IMU_n_smooth = cst.IMU_n_smoooth           # 5 + 1 + 5 frames running average, past to future
        self.win_l = cst.ACC_MOVING_AVE_LEN
        self.max_input_l = max_input_l

        # postprocessing smoothing filter
        self.coeff = 0.6 ** np.arange(6)[::-1]

    def record_raw_imu(self, cur_imu: np.ndarray):
        # cur_imu (1, 72)
        if len(self.raw_imu_buffer) == 0:
            for i in range(self.IMU_n_smooth):
                self.raw_imu_buffer.append(cur_imu.copy())

        # so when first time call, we have self.IMU_n_smooth(5) + 1 frames of the same imu
        self.raw_imu_buffer.append(cur_imu.copy())

        if len(self.raw_imu_buffer) >= self.win_l:
            win = np.array(self.raw_imu_buffer[-self.win_l:])
            smoothed = np.concatenate((
                self.raw_imu_buffer[-self.IMU_n_smooth-1][: 6 * 9],
                np.mean(win[:, 6 * 9: 6 * 9 + 18], axis=0),
            ))
            self.smoothed_imu_buffer.append(smoothed)

            assert len(self.smoothed_imu_buffer) == len(self.raw_imu_buffer) - 2 * self.IMU_n_smooth

    def record_state_aa_and_c(self, cur_s: np.ndarray, cur_c: np.ndarray):

        assert cur_s.shape[0] == cst.n_dofs * 2
        s_and_c = np.concatenate((
            batch_to_rot_mat_2axis((cur_s[3: cst.n_dofs + 3])[np.newaxis, :])[0],
            cur_c,
        ))
        self.s_and_c_in_buffer.append(s_and_c)

    def smooth_and_split_s_c(self, st_2axis_and_c):

        # for some reason the raw output poses from Transformer are noisy, need a post filter

        self.s_c_smooth_buffer.append(st_2axis_and_c)

        win_l = len(self.coeff)
        if len(self.s_c_smooth_buffer) >= win_l:
            s_smooth = np.array(self.s_c_smooth_buffer[-win_l:]) * self.coeff[:, np.newaxis]
            s_smooth = np.sum(s_smooth, axis=0) / np.sum(self.coeff)
            # c_t = self.s_c_smooth_buffer[-4][-8:]
        else:
            s_smooth = st_2axis_and_c
            # c_t = st_2axis_and_c[-8:]

        # st_2axis_and_c 1D
        st_2axis = s_smooth[:-self.n_sbps*4]
        c_t = s_smooth[-self.n_sbps*4:]

        confidences = c_t[0::4].copy()
        c_t[0::4] = (c_t[0::4] > 0.0) * 1.0
        c_t[1::4] /= 5.0
        c_t[2::4] /= 5.0
        c_t[3::4] /= 5.0

        return st_2axis, c_t, confidences

    def step(
        self, cur_imu: np.ndarray,
        prev_root_xyz: np.ndarray,
    ) -> Dict:
        # input: current imu reading t (72,), previous root xyz (3,)
        # output: state prediction at t - IMU_n_smooth (2 * N_DOFS, )
        # also output global SBP locations for rendering/debugging

        self.record_raw_imu(cur_imu)

        # first self.IMU_n_smooth IMU readings, do nothing
        if len(self.smoothed_imu_buffer) < 1:
            return {"qdq": self.s_init,
                    "viz_locs": np.ones((5, 3)) * 100.0,
                    "ct": np.zeros(self.n_sbps * 4)}

        assert len(self.s_and_c_in_buffer) == len(self.smoothed_imu_buffer)
        in_imu = np.array(self.smoothed_imu_buffer[-self.max_input_l:])     # max length 40
        in_imu = imu_rotate_to_local(in_imu)

        if self.with_acc_sum:
            # after rotation sum
            in_imu_acc_sum = np.sum(in_imu[-cst.ACC_SUM_WIN_LEN:, 54:72], axis=0)   # max sum over 40 steps.
            self.imu_acc_sum_buffer.append(in_imu_acc_sum)
            assert len(self.smoothed_imu_buffer) == len(self.imu_acc_sum_buffer)
            in_imu_acc_sum_window = np.array(self.imu_acc_sum_buffer[-self.max_input_l:])     # max length 40
            in_imu_acc_sum_window /= cst.ACC_SUM_DOWN_SCALE
            in_imu = np.concatenate((in_imu, in_imu_acc_sum_window), axis=1)

        len_imu = in_imu.shape[0]
        in_s_and_c = np.array(self.s_and_c_in_buffer[-len_imu:])

        x_imu = torch.tensor(in_imu).float().unsqueeze(0)
        x_s_and_c = torch.tensor(in_s_and_c).float().unsqueeze(0)

        y = self.model(x_imu.cuda(), x_s_and_c.cuda()).cpu()
        st_2axis_root_v_and_c = y.squeeze(0)[-1, :].detach().numpy()

        st_2axis_root_v, c_t, confs = self.smooth_and_split_s_c(st_2axis_root_v_and_c)

        root_v = st_2axis_root_v[-3:]
        st_aa = batch_rot_mat_2axis_to_aa(st_2axis_root_v[:-3][np.newaxis, :])[0]

        s_t = self.s_init.copy() * 0.0
        s_t[cst.n_dofs: cst.n_dofs + 3] = root_v        # not used for later steps though
        s_t[:3] = prev_root_xyz + root_v * cst.DT
        s_t[6:cst.n_dofs] = st_aa[3:]       # ignore root rotation prediction, provided by IMU directly
        A = conversions.R2A(np.reshape(in_imu[-1, :9], (3, 3)))
        s_t[3:6] = A

        # To make motion a bit smoother, can be safely removed.
        if self.last_s is not None:
            s_t[6:] = (s_t[6:] + self.last_s[6:]) / 2.0
        self.last_s = s_t.copy()

        s_t_bullet = our_pose_2_bullet_format(self.char, s_t)
        pq_g, pq_g_jf = viz_current_frame_and_store_fk_info_include_fixed(
            self.char, s_t_bullet, return_joint_frame_info=True
        )
        pg_prev = self.pq_g_buffer[-1]

        # vel_res, self.c_locs = get_cur_step_root_correction_from_feet_constr(
        #     self.char, pg_prev, pq_g, c_t[:8], DT * 1.0
        # )
        # # since we only used 2 SBPs, need 3 dummy ones to make self.c_locs always same size
        # self.c_locs = np.concatenate((self.c_locs, np.ones((3, 3)) * 100.0), axis=0)
        vel_res, self.c_locs, raw_v_residues = get_cur_step_root_correction_from_all_constr(
            self.char, pg_prev, pq_g, c_t, cst.DT * 1.0, use_n_sbps=np.minimum(5, self.n_sbps)
        )

        # assume ground flat
        vel_res[2] = 0.0
        if np.linalg.norm(self.c_locs[0]) < 100:
            vel_res[2] += self.c_locs[0, 2] * 1.0
        if np.linalg.norm(self.c_locs[1]) < 100:
            vel_res[2] += self.c_locs[1, 2] * 1.0

        self.c_locs = self.c_locs - vel_res * cst.DT
        s_t[:3] -= vel_res * cst.DT
        pq_g[:, :3] -= vel_res[np.newaxis, :] * cst.DT
        self.pq_g_buffer.append(pq_g)

        self.record_state_aa_and_c(s_t, c_t)

        return {"qdq": np.array(s_t),
                "viz_locs": self.c_locs,  # broadcast
                "ct": c_t}
