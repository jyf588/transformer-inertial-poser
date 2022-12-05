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
    imu_rotate_to_local, leg_two_joint_ik_keep_foot_pointing, two_joint_ik
import constants as cst


def is_c_loc_active(c_loc):
    # if not active, default value of something close to (100, 100, 100)
    return np.linalg.norm(c_loc) < 100


class RTRunner:
    def __init__(
        self,
        char: SimAgent,
        model_kin: torch.nn.Module,
        max_input_l: int,
        s_init: np.array,
        map_bound: float,
        grid_size: float,
        play_back_gt=False,
        five_sbp=False,
        with_acc_sum=False,
        multi_sbp_terrain_and_correction=False
    ):
        self.play_back_gt = play_back_gt
        self.n_sbps = 5 if five_sbp else 2
        self.with_acc_sum = with_acc_sum
        self.multi_sbp_terrain_and_correction = multi_sbp_terrain_and_correction

        self.model = model_kin
        self.char = char

        self.s_and_c_in_buffer = []     # (18*6 + 3 + n_sbps * 4)
        self.raw_imu_buffer = []        # (72)
        self.smoothed_imu_buffer = []   # (72)
        self.imu_acc_sum_buffer = []   # (18)
        self.s_c_smooth_buffer = []     # (18*6 + 3 + n_sbps * 4)
        self.pq_g_buffer = []           # history list of global pos&ori of all bodies

        self.c_locs = np.ones((self.n_sbps, 3)) * 100.0
        self.c_locs_prev = self.c_locs.copy()

        self.sbp_idx = {
            "lankle": 0,
            "rankle": 1,
            "lwrist": 2,
            "rwrist": 3,
            "root": 4
        }

        self.ik_target_deltas = {
            "lankle": np.zeros(3),
            "rankle": np.zeros(3),
            "lwrist": np.zeros(3),
            "rwrist": np.zeros(3),
            "root": np.zeros(3)
        }

        self.establishing_height_phase_tick = {
            "lankle": -1,
            "rankle": -1,
            "root": -1
        }
        self.establishing_height_phase_max_len = 50     # 50 steps, long enough..

        # parent link and limb links
        # check amass_char_info
        self.ik_chain_map_bullet = {
            "lankle": [-1, 0, 1, 2],
            "rankle": [-1, 3, 4, 5],
            "lwrist": [11, 12, 13, 14],
            "rwrist": [15, 16, 17, 18],
        }

        # limb joint angles to change
        # check amass_char_info: nimble_state_map
        self.ik_chain_map_nimble_joint = {
            "lankle": [1, 2, 3],
            "rankle": [15, 16, 17],
            "lwrist": [8, 9],       # wrist joint fixed, only change shoulder and elbow
            "rwrist": [13, 14],
        }

        self.s_init = s_init            # (2 * N_DOFS, ) (q, dq), dq not predicted except root vel though
        self.last_s = None              # (2 * N_DOFS, ) (q, dq), dq not predicted except root vel though
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

        self.map_bound = map_bound           # [-map_bound m, map_bound m]
        self.grid_size = grid_size
        self.grid_num = int(self.map_bound/self.grid_size) * 2

        # self.height_map = np.ones((self.grid_num, self.grid_num)) * -0.0

        self.height_region_map = np.zeros((self.grid_num, self.grid_num), dtype=int)
        self.height_confidence_map = np.ones((self.grid_num, self.grid_num)) * -100.0
        self.region_height_list = [0.0]
        self.region_weight_list = [10.0]
        self.temporal_inertial = 1.0            # might need to tune this
        self.height_correction_force = 20.0     # might need to tune this
        self.pelvis_terrain_thres = 0.2        # might need to tune this

        # 1m * 1m square region for now. maybe circular region better.
        self.diffuse_region = round(0.5 / self.grid_size)

        def gen_confidence_region_map():
            d = self.diffuse_region
            x = np.arange(-d, d)
            y = np.arange(-d, d)
            xx, yy = np.meshgrid(x, y)
            zz = - np.sqrt(xx ** 2 + yy ** 2)     # cost func probably no matter since we use for ranking only
            return zz
        self.diffuse_confidence_map = gen_confidence_region_map()       # 20 x 20
        self.update_epsilon = 0.1      # do not update terrain if difference too small, might need to tune this

    def update_height_map_new(self, link: str) -> float:
        # return the height amount need to be corrected, usually (height map height - sbp height)
        # print(self.region_height_list)

        def loc_xy_to_grid_idx(loc: Tuple[float, float]) -> Tuple[int, int]:
            x, y = loc
            return round(x/self.grid_size) + self.grid_num//2, round(y/self.grid_size) + self.grid_num//2

        def update_height_region_map_and_confidence_map(
                update_region_idx: int,
        ):
            nonlocal region_old, confidence_old

            region_new = np.ones_like(region_old) * update_region_idx
            confidence_new = self.diffuse_confidence_map.copy()

            region_merge = np.where(
                confidence_old > confidence_new,       # keep old terrain if == ?
                region_old,
                region_new
            )
            confidence_merge = np.maximum(confidence_old, confidence_new)

            self.height_confidence_map[
                update_boundaries[0]: update_boundaries[1],
                update_boundaries[2]: update_boundaries[3]
            ] = confidence_merge
            self.height_region_map[
                update_boundaries[0]: update_boundaries[1],
                update_boundaries[2]: update_boundaries[3]
            ] = region_merge

            return

        def check_neighboring_area_for_similar_height_region(height_query: float) -> int:
            # return true or false enough?
            # doesn't matter where do we find the similar-height grid nearby
            # if ture, update cluster height a bit
            # else, create a new region/cluster

            nonlocal region_old
            region_choices = list(set(region_old.flatten()))

            if height_query < self.region_height_list[0] + self.update_epsilon:
                return 0

            heights = []
            for choice in region_choices:
                heights.append(self.region_height_list[choice])
            heights = np.array(heights)

            # when multiple close cluster, should just choose spatially close one?
            diffs = np.abs(heights - height_query)
            if np.min(diffs) < self.update_epsilon:
                return region_choices[np.argmin(diffs)]
            else:
                # nothing nearby patches of similar height
                return -1

        # basically the same as using c_locs. need c_locs_prev as we need do a final update the first step SBP is off.
        this_c_loc = self.c_locs_prev[self.sbp_idx[link]]
        if not is_c_loc_active(this_c_loc):
            return 0

        tick = self.establishing_height_phase_tick[link]
        if tick < 0:
            self.establishing_height_phase_tick[link] = self.establishing_height_phase_max_len  # wait
            return 0
        elif tick > 0:
            return 0  # waiting...

        # tick == 0, do update!
        update_height = this_c_loc[2]
        update_center = (this_c_loc[0], this_c_loc[1])
        center_idx = loc_xy_to_grid_idx(update_center)
        # print(update_center)
        # print(center_idx)
        # center_old_region_idx = self.height_region_map[center_idx]

        update_boundaries = [
            center_idx[0] - self.diffuse_region,
            center_idx[0] + self.diffuse_region,
            center_idx[1] - self.diffuse_region,
            center_idx[1] + self.diffuse_region
        ]
        region_old = self.height_region_map[
                     update_boundaries[0]: update_boundaries[1],
                     update_boundaries[2]: update_boundaries[3]
                     ].copy()
        confidence_old = self.height_confidence_map[
                         update_boundaries[0]: update_boundaries[1],
                         update_boundaries[2]: update_boundaries[3]
                         ].copy()

        center_new_region_idx = check_neighboring_area_for_similar_height_region(update_height)

        if center_new_region_idx < 0:
            # need to create a new region
            center_new_region_idx = len(self.region_height_list)
            self.region_height_list.append(update_height)
            self.region_weight_list.append(10.0)            # TODO
        else:
            old_h = self.region_height_list[center_new_region_idx]
            old_w = self.region_weight_list[center_new_region_idx]

            # more points in this region means more confidence
            self.region_weight_list[center_new_region_idx] += 1.0
            # update region height with inertia
            self.region_height_list[center_new_region_idx] = \
                (old_h * old_w * self.temporal_inertial + update_height * 1.0) / (old_w * self.temporal_inertial + 1.0)

        self.establishing_height_phase_tick[link] = -1

        update_height_region_map_and_confidence_map(center_new_region_idx)

        # mid_p = self.height_region_map.shape[0] // 2
        # print(repr(self.height_region_map[mid_p-20:mid_p+20, mid_p-20:mid_p+20]))
        # print(repr(self.height_confidence_map[mid_p-20:mid_p+20, mid_p-20:mid_p+20]))
        # print(self.height_confidence_map[center_idx])
        # print(self.height_region_map[center_idx])

        # return self.region_height_list[center_new_region_idx] - update_height
        return self.region_height_list[self.height_region_map[center_idx]] - update_height

    def update_sbp_establishing_height_ticks(self):
        links = ['lankle', 'rankle', 'root']
        for link in links:
            assert self.establishing_height_phase_tick[link] >= -1
            if self.establishing_height_phase_tick[link] < 0:
                continue
            else:
                self.establishing_height_phase_tick[link] -= 1

            this_c_loc = self.c_locs[self.sbp_idx[link]]
            prev_c_loc = self.c_locs_prev[self.sbp_idx[link]]
            if (not is_c_loc_active(this_c_loc)) and is_c_loc_active(prev_c_loc):
                # end establishing phase immediately
                self.establishing_height_phase_tick[link] = 0

    def record_raw_imu(self, cur_imu: np.ndarray):
        # cur_imu (1, 72)
        if len(self.raw_imu_buffer) == 0:
            for i in range(self.IMU_n_smooth):
                self.raw_imu_buffer.append(cur_imu.copy())

        self.raw_imu_buffer.append(cur_imu.copy())

        if len(self.raw_imu_buffer) >= self.win_l:
            win = np.array(self.raw_imu_buffer[-self.win_l:])
            smoothed = np.concatenate((
                self.raw_imu_buffer[-self.IMU_n_smooth-1][: 6 * 9],
                np.mean(win[:, 6 * 9: 6 * 9 + 18], axis=0),
                # self.raw_imu_buffer[-self.IMU_n_smooth-1][6 * 9: ],
            ))
            self.smoothed_imu_buffer.append(smoothed)

            assert len(self.smoothed_imu_buffer) == len(self.raw_imu_buffer) - 2*self.IMU_n_smooth

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

    def correct_joint_q_for_history_feedback(
        self,
        s4hist: np.ndarray,
        pq_jf_cur: np.ndarray,
        raw_v_residues: np.ndarray,
        link: str
    ) -> np.ndarray:

        root_v_residue = raw_v_residues[self.sbp_idx["root"], :]
        sbp_v_residue = raw_v_residues[self.sbp_idx[link], :]       # non-root

        i_p, i_a, i_b, i_c = tuple(self.ik_chain_map_bullet[link])
        pq_parent, pq_a, pq_b, pq_c, = pq_jf_cur[i_p + 1], pq_jf_cur[i_a + 1], pq_jf_cur[i_b + 1], pq_jf_cur[i_c + 1]

        if not np.isnan(sbp_v_residue).all() and not np.isnan(root_v_residue).all():
            self.ik_target_deltas[link] += (sbp_v_residue - root_v_residue) * cst.DT
            # print(self.ik_target_deltas[link])
            correction_vec = -self.ik_target_deltas[link]

            if np.linalg.norm(correction_vec) > 0.5:
                self.ik_target_deltas[link] = np.zeros(3)
                return s4hist       # do nothing

            # if cond probably unnecessary (avoid zero norm quat)
            if np.linalg.norm(correction_vec) > 0.05:
                if link == "lankle" or link == "rankle":
                    a_q_l_1, b_q_l_1, c_q_l_1 = leg_two_joint_ik_keep_foot_pointing(
                        pq_parent, pq_a, pq_b, pq_c, correction_vec
                    )
                    a_l_1_aa = conversions.Q2A(a_q_l_1)
                    b_l_1_aa = conversions.Q2A(b_q_l_1)
                    c_l_1_aa = conversions.Q2A(c_q_l_1)
                    j_a, j_b, j_c = tuple(self.ik_chain_map_nimble_joint[link])
                    s4hist[3 + j_a * 3: 6 + j_a * 3] = a_l_1_aa  # hip
                    s4hist[3 + j_b * 3: 6 + j_b * 3] = b_l_1_aa  # knee
                    s4hist[3 + j_c * 3: 6 + j_c * 3] = c_l_1_aa  # ankle
                elif link == "lwrist" or link == "rwrist":
                    a_q_l_1, b_q_l_1 = two_joint_ik(
                        pq_parent, pq_a, pq_b, pq_c, correction_vec, is_arm=True
                    )
                    a_l_1_aa = conversions.Q2A(a_q_l_1)
                    b_l_1_aa = conversions.Q2A(b_q_l_1)
                    j_a, j_b = tuple(self.ik_chain_map_nimble_joint[link])
                    s4hist[3 + j_a * 3: 6 + j_a * 3] = a_l_1_aa  # shoulder
                    s4hist[3 + j_b * 3: 6 + j_b * 3] = b_l_1_aa  # elbow
        else:
            self.ik_target_deltas[link] = np.zeros(3)

        return s4hist

    def step(
        self, cur_imu: np.ndarray,
        prev_root_xyz: np.ndarray,
        t: int,
        s_gt: Union[np.ndarray, None] = None,
        c_gt: Union[np.ndarray, None] = None
    ) -> Dict:
        # input: current imu reading t (72,), previous root xyz (3,)
        # output: state prediction at t - IMU_n_smooth (2 * N_DOFS, )
        # also output global SBP locations for rendering/debugging

        # conf_l, conf_r = 100.0, 100.0
        if self.play_back_gt:

            s_t = s_gt[t].copy()
            # s_t[:3] = prev_root_xyz + s_t[N_DOFS:N_DOFS+3] * DT
            c_t = c_gt[t]
            confs = np.ones(self.n_sbps) * 100.0
        else:
            self.record_raw_imu(cur_imu)

            # first self.IMU_n_smooth steps
            if len(self.smoothed_imu_buffer) < 1:
                return {"qdq": self.s_init,
                        "viz_locs": np.ones((5, 3)) * 100.0,
                        "ct": np.zeros(self.n_sbps * 4)}

            assert len(self.s_and_c_in_buffer) == len(self.smoothed_imu_buffer)

            in_imu = np.array(self.smoothed_imu_buffer[-self.max_input_l:])     # max length 40
            in_imu = imu_rotate_to_local(in_imu)

            if self.with_acc_sum:
                # after rotation sum
                in_imu_acc_sum = np.sum(in_imu[-cst.ACC_SUM_WIN_LEN:, 54:72], axis=0)       # max sum over 40 steps.
                self.imu_acc_sum_buffer.append(in_imu_acc_sum)
                assert len(self.smoothed_imu_buffer) == len(self.imu_acc_sum_buffer)
                in_imu_acc_sum_window = np.array(self.imu_acc_sum_buffer[-self.max_input_l:])     # max length 40
                in_imu_acc_sum_window /= cst.ACC_SUM_DOWN_SCALE
                in_imu = np.concatenate((in_imu, in_imu_acc_sum_window), axis=1)

            len_imu = in_imu.shape[0]
            in_s_and_c = np.array(self.s_and_c_in_buffer[-len_imu:])

            x_s_and_c = torch.tensor(in_s_and_c).float().unsqueeze(0)
            x_imu = torch.tensor(in_imu).float().unsqueeze(0)

            y = self.model(x_imu.cuda(), x_s_and_c.cuda()).cpu()
            st_2axis_root_v_and_c = y.squeeze(0)[-1, :].detach().numpy()

            st_2axis_root_v, c_t, confs = self.smooth_and_split_s_c(st_2axis_root_v_and_c)

            root_v = st_2axis_root_v[-3:]
            st_aa = batch_rot_mat_2axis_to_aa(st_2axis_root_v[:-3][np.newaxis, :])[0]

            s_t = self.s_init.copy() * 0.0
            s_t[cst.n_dofs: cst.n_dofs + 3] = root_v  # not used for later steps though
            s_t[:3] = prev_root_xyz + root_v * cst.DT
            s_t[6: cst.n_dofs] = st_aa[3:]  # ignore root rotation prediction, provided by IMU directly
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
        self.pq_g_buffer.append(pq_g)

        pg_prev = self.pq_g_buffer[-2]
        vel_res, self.c_locs, raw_v_residues = get_cur_step_root_correction_from_all_constr(
            self.char, pg_prev, pq_g, c_t, cst.DT * 1.0, use_n_sbps=np.minimum(5, self.n_sbps)
        )

        vel_res[2] = 0.0                        # use terrain to correct root height, not SBPs
        self.c_locs = self.c_locs - vel_res * cst.DT

        self.update_sbp_establishing_height_ticks()

        d = self.update_height_map_new("lankle")
        # print("l", d, self.establishing_height_phase_tick['lankle'])
        vel_res[2] += -d * self.height_correction_force
        d = self.update_height_map_new("rankle")
        # print("r", d, self.establishing_height_phase_tick['rankle'])
        vel_res[2] += -d * self.height_correction_force

        # hacky
        dist = np.linalg.norm(
            pq_g[self.char.get_char_info().root + 1][:2] - \
            (pq_g[self.char.get_char_info().lankle + 1][:2] + pq_g[self.char.get_char_info().rankle + 1][:2]) / 2
        )
        if self.multi_sbp_terrain_and_correction and dist > self.pelvis_terrain_thres:
            _ = self.update_height_map_new("root")

        ##########
        st_hist_copy = s_t.copy()
        if self.multi_sbp_terrain_and_correction:
            st_hist_copy = self.correct_joint_q_for_history_feedback(st_hist_copy, pq_g_jf, raw_v_residues, "lankle")
            st_hist_copy = self.correct_joint_q_for_history_feedback(st_hist_copy, pq_g_jf, raw_v_residues, "rankle")
        ##########

        if not self.play_back_gt:
            s_t[:3] = s_t[:3] - vel_res * cst.DT
            st_hist_copy[:3] = st_hist_copy[:3] - vel_res * cst.DT      # actually not needed since root not part of hist
            for pq in self.pq_g_buffer[-1]:
                pq[:3] = pq[:3] - vel_res * cst.DT

        self.record_state_aa_and_c(st_hist_copy, c_t)       # not s_t itself, but the corrected version
        self.c_locs_prev = self.c_locs.copy()

        return {"qdq": np.array(s_t),
                "viz_locs": self.c_locs,  # broadcast
                "ct": c_t}
