# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) Stanford University
# all util functions not for training.

import pickle
import time
from typing import List, Tuple, Union

import numpy as np
import pybullet as pb
import torch
from fairmotion.core.motion import Motion
from fairmotion.ops import conversions, quaternion
from fairmotion.ops.quaternion import Q_mult
from einops import rearrange

from bullet_agent import SimAgent
import constants as cst


def draw_ori(pb_client, R):
    pb_client.addUserDebugLine((0, 0, 0), list(R[:, 0]), (1, 0, 0), 5)
    pb_client.addUserDebugLine((0, 0, 0), list(R[:, 1]), (0, 1, 0), 5)
    pb_client.addUserDebugLine((0, 0, 0), list(R[:, 2]), (0, 0, 1), 5)


def get_rot_center_sample_based(x1, q1, x2, q2, dt, sol_prev, link):
    v = (x2 - x1) / dt

    # if np.linalg.norm(v) < 0.5:
    #     return np.array([0., 0, 0]), v
    # else:
    #     return None, np.array([0., 0, 0])

    sub = q2 - q1 if np.linalg.norm(q2 - q1) < np.linalg.norm(q2 + q1) else q2 + q1
    dori = 2 * Q_mult(sub, q2 * np.array([-1., -1, -1, 1]))
    w = (dori / dt)[:3]

    # Q_diff = Q_mult(q2, q1 * np.array([-1., -1, -1, 1]))
    # Q_diff = -Q_diff if Q_diff[3] < 0.0 else Q_diff
    # wq = conversions.Q2A(Q_diff) / dt
    # print(w, wq)

    #
    # Q_diff = Q_mult(q1 * np.array([-1., -1, -1, 1]), q2)
    # w1 = conversions.Q2A(Q_diff) / dt
    # w1r = conversions.Q2R(q2).dot(w1)
    # w1q = Q_mult(Q_mult(q2, np.concatenate((w1, [0]))), q2 * np.array([-1., -1, -1, 1]))[:3]
    #
    # print(w, w1r, w1q)      # all equivalent

    if link == 14 or link == 18:  # ref amass_char_info
        # left or right wrist
        lp_x = np.arange(-0.02, 0.03, 0.01)
        lp_y = np.arange(-0.02, 0.03, 0.01)
        lp_z = np.arange(-0.02, 0.03, 0.01)
    elif link == 2 or link == 5:
        # left or right foot
        lp_x = np.arange(-0.04, 0.05, 0.01)
        lp_y = np.arange(-0.04, 0.02, 0.01)
        lp_z = np.arange(-0.15, 0.18, 0.01)  # size 11
    elif link == -1:
        # pelvis
        lp_x = np.arange(-0.15, 0.16, 0.01)
        lp_y = np.arange(-0.1, 0.15, 0.01)
        lp_z = np.arange(-0.12, -0.04, 0.01)
    else:
        assert False

    xx, yy, zz = np.meshgrid(lp_x, lp_y, lp_z)
    lps = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1)

    r2 = conversions.Q2R(q2)
    lps_R = np.einsum('ij,bj->bi', r2, lps)     # R*p

    w1, w2, w3 = tuple(w)
    wx = np.array([[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]])

    wx_lps_R = np.einsum('ij,bj->bi', wx, lps_R)

    # v_along_w = get_projection_vec(v, w)
    # v_orth = v - v_along_w
    # lps_v = wx_lps_R + v_orth[np.newaxis, :]

    lps_v = wx_lps_R + v[np.newaxis, :]

    if sol_prev is None:
        dist = np.zeros(lps.shape)
    else:
        dist = lps_R - (sol_prev - v * dt)[np.newaxis, :]
        # dist = lps_R - sol_prev[np.newaxis, :]

    residues_v = np.linalg.norm(lps_v, axis=1)
    residues = residues_v + 0.2 * np.linalg.norm(dist, axis=1) + 0.02 * np.linalg.norm(lps_R, axis=1)
    idx = np.argmin(residues)

    if residues[idx] < cst.V_THRES:
        return lps_R[idx], lps_v[idx]
    else:
        return None, np.array([0., 0, 0])


def get_raw_motion_info_nimble_q_dummy_dq(
        char: SimAgent,
        m: Motion,
) -> np.ndarray:
    # a matrix of size l-by-(x, y ,z, root_aa, j_aa, vx, vy, vz, root_w, j_w)
    # use NimblePhysics order
    # note that dq, except root velocity, is not used in the training, so zeros are filled

    raw_info = []
    cur_time = 0.015 / 2.0
    char_info = char.get_char_info()
    all_joint_idx = char_info.joint_idx.values()

    n_j = len(all_joint_idx) - 1    # should be 19 for verification, exclude root

    while cur_time < m.length():
        cur_pose = m.get_pose_by_time(cur_time)
        j_q = np.zeros(n_j * 3)

        for idx in all_joint_idx:
            if idx == char_info.root:
                continue
            nimble_idx = char_info.nimble_map[idx] - 1
            if char.get_joint_type(idx) == pb.JOINT_FIXED:
                # skip fixed wrist joints
                j_q[nimble_idx * 3: nimble_idx * 3 + 3] = np.nan
            else:
                # all others are spherical
                T = cur_pose.get_transform(char_info.bvh_map[idx], local=True)
                Q, _ = conversions.T2Qp(T)
                A = conversions.Q2A(Q).tolist()
                j_q[nimble_idx * 3: nimble_idx * 3 + 3] = A

        not_nan_array = ~ np.isnan(j_q)
        j_dq = j_q * 0.0      # since the method does not use velocity, just fill with zeros
        j_q_filtered = j_q[not_nan_array].tolist()
        j_dq_filtered = j_dq[not_nan_array].tolist()

        T_root = cur_pose.get_transform(char_info.bvh_map[char_info.ROOT], local=False)
        Q, p = conversions.T2Qp(T_root)

        next_pose = m.get_pose_by_time(cur_time + cst.DT)
        T_root_next = next_pose.get_transform(char_info.bvh_map[char_info.ROOT], local=False)
        Q_n, p_n = conversions.T2Qp(T_root_next)

        v = (p_n - p) / cst.DT
        Q_diff = Q_mult(Q * np.array([-1., -1, -1, 1]), Q_n)
        w = conversions.Q2A(Q_diff) / cst.DT

        cur_info = p.tolist() + conversions.Q2A(Q).tolist() + j_q_filtered + \
            v.tolist() + w.tolist() + j_dq_filtered

        raw_info.append(cur_info)
        cur_time += cst.DT

    raw_info = np.array(raw_info)
    assert raw_info.shape[1] == ((n_j - 2) * 3 + 3 + 3) * 2

    return raw_info


def batch_rot_mat_2axis_to_aa(rm):
    # this func is batched
    # rm: (b, Nj * 6)
    # no root info
    # aa: (b, Nj * 3)
    b = rm.shape[0]
    rm = rearrange(rm, 'b (n r1 r2) -> (b n) r1 r2', r1=3, r2=2)
    a1 = rm[:, :, 0] / (np.linalg.norm(rm[:, :, 0], axis=1, keepdims=True) + 1e-6)
    a2 = rm[:, :, 1] / (np.linalg.norm(rm[:, :, 1], axis=1, keepdims=True) + 1e-6)
    a3 = np.cross(a1, a2)[:, :, np.newaxis]    # b*Nj, 3, 1
    a1 = a1[:, :, np.newaxis]
    a2 = a2[:, :, np.newaxis]
    rm_full = np.concatenate((a1, a2, a3), axis=2)    # b*Nj, 3, 3
    aa = conversions.R2A(rm_full)
    aa = rearrange(aa, '(b n) a -> b (n a)', a=3, b=b)      # (b, Nj * 3)
    return aa


def batch_to_rot_mat_2axis(batch_s):
    assert batch_s.shape[1] == cst.n_dofs
    aa = batch_s[:, :cst.n_dofs - 3].reshape(-1, 3)     # exclude global xyz
    r = conversions.A2R(aa)[:, :, :2]
    r = r.reshape(-1, 6).reshape(batch_s.shape[0], -1)
    return np.concatenate((r, batch_s[:, -3:]), axis=1)     # padding unchanged global xyz back


def imu_rotate_to_local(batch_imu):
    batch_imu = batch_imu.copy()

    root_r = batch_imu[:, :9].reshape(-1, 3, 3)
    other_r = batch_imu[:, 9: 6 * 9].reshape(-1, 5, 3, 3)

    other_r_local = other_r.copy()
    for i in range(5):
        other_r_local[:, i, :, :] = np.matmul(
            np.linalg.inv(root_r),
            other_r[:, i, :, :]
        )

    root_acc = batch_imu[:, 6 * 9: 6 * 9 + 3]
    other_acc = batch_imu[:, 6 * 9 + 3:].reshape(-1, 5, 3)
    other_acc_local = other_acc.copy()
    for i in range(5):
        other_acc_local[:, i, :] = np.einsum(
            'ijk,ik->ij',
            np.linalg.inv(root_r),
            other_acc[:, i, :]
        )

    res = np.concatenate((
        root_r.reshape(-1, 9), other_r_local.reshape(-1, 45),
        root_acc, other_acc_local.reshape(-1, 15)
    ), axis=1)

    assert batch_imu.shape == res.shape
    return res


def dip_pose_2_bullet_format(
        char: SimAgent,
        pose: np.ndarray,
        tran=None
) -> np.ndarray:
    # dip pose (24, 3, 3)
    # optionally dip translation (3, )

    if tran is None:
        tran = np.array([0., 0, 0])

    bullet_q = []
    aa_root = conversions.R2A(cst.rot_up_R.dot(pose[0, :, :]))
    tran = cst.rot_up_R.dot(tran)
    tran[2] += cst.root_z_offset
    bullet_q += list(tran) + list(aa_root)

    for idx in char.non_root_active_idx:
        j_r = pose[cst.SMPL_JOINT_IDX_MAPPING[char.get_char_info().bvh_map[idx]]]
        bullet_q += list(conversions.R2A(j_r))

    return np.array(bullet_q)


def our_pose_2_bullet_format(
        char: SimAgent,
        s_np: np.ndarray
) -> np.ndarray:
    bullet_q = []
    bullet_q += list(s_np[:6])

    for idx in char.non_root_active_idx:
        nimble_state_start = (char.get_char_info().nimble_state_map[idx] - 1) * 3 + 6
        aa = s_np[nimble_state_start:(nimble_state_start + 3)]
        bullet_q += list(aa)

    assert len(bullet_q) == len(s_np) // 2
    return np.array(bullet_q)


def viz_current_frame_and_store_fk_info_include_fixed(
        char: SimAgent,
        state_bullet: np.ndarray,
        return_joint_frame_info=False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    # use the bullet character for both visualization and FK

    state_pos = []
    state_pq_g = []
    state_pq_g_jf = []

    v = w = np.array([0., 0., 0.])
    Q_root = conversions.A2Q(state_bullet[3:6])
    char.set_root_pQvw(state_bullet[:3], Q_root, v, w)

    for i, j_idx in enumerate(char.non_root_active_idx):
        aa = state_bullet[i*3+6: i*3+9]
        state_pos.append(conversions.A2Q(aa))

    char.set_joints_pv(
        char.non_root_active_idx,
        state_pos,
        [np.zeros(3)] * len(char.non_root_active_idx)
    )

    root_p, root_q = char.get_root_pQ()
    state_pq_g.append(list(root_p) + list(root_q))

    if return_joint_frame_info:
        state_pq_g_jf.append(list(root_p) + list(root_q))

    non_root_all_idx = list(char._joint_indices)
    links_p, links_q = char.get_link_pQ(non_root_all_idx)
    for p, q in zip(links_p, links_q):
        state_pq_g.append(list(p) + list(q))

    if return_joint_frame_info:
        links_p_jf, links_q_jf = char.get_link_pQ_joint_frame(non_root_all_idx)
        for p, q in zip(links_p_jf, links_q_jf):
            state_pq_g_jf.append(list(p) + list(q))

    if return_joint_frame_info:
        return np.array(state_pq_g), np.array(state_pq_g_jf)
    else:
        return np.array(state_pq_g)


"""
    metrics to report
"""


def loss_angle(aa_1, aa_2, pq_g_1, pq_g_2):
    aa_1 = aa_1[:, 3:].reshape(-1, 3)
    aa_2 = aa_2[:, 3:].reshape(-1, 3)     # first 3 global trans, not rot aa

    diff_qs = quaternion.Q_diff(conversions.A2Q(aa_1), conversions.A2Q(aa_2))
    diff_qs = diff_qs * np.sign(diff_qs[:, 3:4])
    diff_angles = np.linalg.norm(conversions.Q2A(diff_qs), axis=1)

    # aa_1 = torch.from_numpy(aa_1)
    # aa_2 = torch.from_numpy(aa_2)
    # diff_aa = axisangle_diff_torch(aa_1, aa_2)
    # diff_angles = np.linalg.norm(diff_aa.numpy(), axis=1)

    return diff_angles.mean() * 180 / 3.1416


def loss_j_pos(aa_1, aa_2, pq_g_1, pq_g_2):

    p_g_1 = pq_g_1[:, 1:, :3] - pq_g_1[:, 0:1, :3]      # (time, joint, pq), in root coord
    p_g_2 = pq_g_2[:, 1:, :3] - pq_g_2[:, 0:1, :3]      # (time, joint, pq), in root coord
    p_g_1 = p_g_1.reshape(-1, 3)
    p_g_2 = p_g_2.reshape(-1, 3)
    diff_p = np.linalg.norm(p_g_2 - p_g_1, axis=1)
    return diff_p.mean() * 100.0            # m to cm


def loss_global_angle(aa_1, aa_2, pq_g_1, pq_g_2):

    q_g_1 = pq_g_1[:, :, 3:]     # (time, joint, pq), in root coord
    q_g_2 = pq_g_2[:, :, 3:]      # (time, joint, pq), in root coord
    q_g_1 = q_g_1.reshape(-1, 4)
    q_g_2 = q_g_2.reshape(-1, 4)

    diff_qs = quaternion.Q_diff(q_g_1, q_g_2)
    diff_qs = diff_qs * np.sign(diff_qs[:, 3:4])
    diff_angles = np.linalg.norm(conversions.Q2A(diff_qs), axis=1)

    # aa_g_1 = torch.from_numpy(conversions.Q2A(q_g_1))
    # aa_g_2 = torch.from_numpy(conversions.Q2A(q_g_2))
    # diff_aa = axisangle_diff_torch(aa_g_1, aa_g_2)
    # diff_angles = np.linalg.norm(diff_aa.numpy(), axis=1)

    return diff_angles.mean() * 180 / 3.1416


def loss_max_jerk(aa_1, aa_2, pq_g_1, pq_g_2):
    # traj 2 is predicted

    # (time-3, joint, p) in global frame
    p_g_2 = pq_g_2[:, :, :3]
    jerk = p_g_2[3:] - 3 * p_g_2[2:-1] + 3 * p_g_2[1:-2] - p_g_2[:-3]
    jerk_norm = np.linalg.norm(jerk, axis=2)
    # jerk_norm_max = np.max(jerk_norm, axis=1)
    jerk_norm_max = np.mean(jerk_norm, axis=1)
    return jerk_norm_max.mean() * 100.0


def loss_root_jerk(aa_1, aa_2, pq_g_1, pq_g_2):
    # traj 2 is predicted

    # (time-3, joint, p) in global frame
    p_r_2 = pq_g_2[:, 0, :3]
    jerk = p_r_2[3:] - 3 * p_r_2[2:-1] + 3 * p_r_2[1:-2] - p_r_2[:-3]
    jerk_norm = np.linalg.norm(jerk, axis=1)
    return jerk_norm.mean() * 100.0


def loss_root_dist_pos(aa_1, aa_2, pq_g_1, pq_g_2, t=1.0):

    ind = int(t / cst.DT) - 1
    max_ind = pq_g_1.shape[0] - 1

    ind = np.minimum(ind, max_ind)

    dxyz_1 = pq_g_1[ind, 0, :3] - pq_g_1[0, 0, :3]
    dxyz_2 = pq_g_2[ind, 0, :3] - pq_g_2[0, 0, :3]

    return np.linalg.norm(dxyz_1 - dxyz_2)


""" metrics to report end """


def get_residue_from_contr(x1, q1, x2, q2, dt, sol, is_local=False):
    v = (x2 - x1) / dt
    sub = q2 - q1 if np.linalg.norm(q2 - q1) < np.linalg.norm(q2 + q1) else q2 + q1
    dori = 2 * quaternion.Q_mult(sub, q2 * np.array([-1., -1, -1, 1]))
    w = (dori / dt)[:3]

    R = conversions.Q2R(q2)
    if is_local:
        sol = R.dot(sol)

    w1, w2, w3 = tuple(w)
    wx = np.array([[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]])
    v_residue = wx.dot(sol) + v

    # print(v, v_residue)
    return v_residue

# def get_cur_step_root_correction_from_feet_constr(
#         char: SimAgent,
#         pq_prev: np.ndarray,
#         pq_cur: np.ndarray,
#         constr_lr: np.ndarray,
#         dt: float
# ) -> (np.ndarray, np.ndarray, np.ndarray):
#
#     # # for debugging viz
#     # viz_loc_l = None
#     # viz_loc_r = None
#
#     import random
#
#     def compute_contr_vel_residue(
#             contr: np.ndarray,
#             link: int,
#     ):
#         xq1 = pq_prev[link + 1]
#         x1, q1 = xq1[:3], xq1[3:]
#
#         xq2 = pq_cur[link + 1]
#         x2, q2 = xq2[:3], xq2[3:]
#
#         if contr[0] == 0.0:
#             viz_loc = np.array([100.0, 100., 100.])     # somewhere far away in GUI
#             residue = np.array([0., 0, 0])
#         else:
#             assert contr[0] == 1.0
#             viz_loc = x2 + contr[1:4]
#             residue = get_residue_from_contr(x1, q1, x2, q2, dt, contr[1:4])
#
#         return residue, viz_loc
#
#     body = char.get_char_info().lankle
#     contr_l = constr_lr[0:4]
#     v_residue_l, viz_loc_l = compute_contr_vel_residue(contr_l, body)
#
#     body = char.get_char_info().rankle
#     contr_r = constr_lr[4:8]
#     v_residue_r, viz_loc_r = compute_contr_vel_residue(contr_r, body)
#
#     # v_residue = v_residue_l
#
#     if np.linalg.norm(v_residue_r) == 0.0:  # no/inactive constraint
#         v_residue = v_residue_l
#     elif np.linalg.norm(v_residue_l) == 0.0:
#         v_residue = v_residue_r
#     else:
#         v_residue = v_residue_l if random.random() > 0.5 else v_residue_r
#         # v_residue = v_residue_l if np.linalg.norm(v_residue_l) > np.linalg.norm(v_residue_r) else v_residue_r
#
#     # TODO: shrug..
#     v_residue = np.clip(v_residue, -0.5, 0.5)
#
#     return v_residue, np.concatenate((viz_loc_l[np.newaxis, :], viz_loc_r[np.newaxis, :]), axis=0)
#


def get_cur_step_root_correction_from_all_constr(
        char: SimAgent,
        pq_prev: np.ndarray,
        pq_cur: np.ndarray,
        constrs: np.ndarray,
        dt: float,
        use_n_sbps=2
) -> (np.ndarray, np.ndarray, np.ndarray):

    def compute_contr_vel_residue(
            contr: np.ndarray,
            link: int,
    ):
        xq1 = pq_prev[link + 1]
        x1, q1 = xq1[:3], xq1[3:]

        xq2 = pq_cur[link + 1]
        x2, q2 = xq2[:3], xq2[3:]

        if contr[0] == 0.0:
            viz_loc = np.array([100.0, 100., 100.])     # somewhere far away in GUI
            residue = np.array([np.nan] * 3)
        else:
            assert contr[0] == 1.0
            viz_loc = x2 + contr[1:4]
            residue = get_residue_from_contr(x1, q1, x2, q2, dt, contr[1:4])

        return residue, viz_loc

    bodies = [
        char.get_char_info().lankle,
        char.get_char_info().rankle,
        char.get_char_info().lwrist,
        char.get_char_info().rwrist,
        char.get_char_info().root,
    ]
    viz_locs = np.ones((5, 3)) * 100.0
    v_residues = np.ones((5, 3)) * np.nan

    for i in range(use_n_sbps):
        body = bodies[i]
        contr_vec = constrs[i * 4:i * 4 + 4]
        v_residues[i, :], viz_locs[i, :] = compute_contr_vel_residue(contr_vec, body)

    if np.isnan(v_residues[:2, :]).all():
        v_residue = np.zeros(3)
    else:
        v_residue = np.nanmean(v_residues[:2, :], axis=0)

    # if np.isfinite(v_residues[2, ]).all() and np.isnan(v_residues[3, ]).all():
    #     v_residue = v_residues[2, ]
    # elif np.isfinite(v_residues[3, ]).all() and np.isnan(v_residues[2, ]).all():
    #     v_residue = v_residues[3, ]
    # elif np.isfinite(v_residues[2, ]).all() and np.isfinite(v_residues[3, ]).all():
    #     v_residue = v_residues[2, ] if random.random() > 0.5 else v_residues[3, ]
    # else:
    #     if np.isnan(v_residues).all():
    #         v_residue = np.zeros(3)
    #     else:
    #         v_residue = np.nanmean(v_residues, axis=0)

    # import random
    # v_residue_l = v_residues[0, :]
    # v_residue_r = v_residues[1, :]
    # if np.isfinite(v_residue_l).all() and np.isfinite(v_residue_r).all():
    #     v_residue = v_residue_l if random.random() > 0.5 else v_residue_r
    # elif np.isfinite(v_residue_l).all():
    #     v_residue = v_residue_l
    # elif np.isfinite(v_residue_r).all():
    #     v_residue = v_residue_r

    # TODO: shrug..
    v_residue = np.clip(v_residue, -0.5, 0.5)

    # also return the raw residues for IK
    return v_residue, viz_locs, v_residues


def normalize(v: np.ndarray) -> np.ndarray:
    v_n = v / (np.linalg.norm(v) + 1e-4)
    return v_n


def two_joint_ik(
    pq_jf_pa, pq_jf_a, pq_jf_b, pq_jf_c, c_delta, is_arm=False
) -> (np.ndarray, np.ndarray):
    # input: joint frame p & q (concatenated) of parent, a(hip), b(knee), and c(ankle)
    # https://theorangeduck.com/page/simple-two-joint
    # a_q_g = quat_mult(pa_q_g, a_q_l), b_q_g = quat_mult(a_q_g, b_q_l)
    # our a_q_g is b_gr in the web link above.
    # world delta vector for c xyz
    # output, new joint angles for a & b in quat.

    a, b, c = pq_jf_a[:3], pq_jf_b[:3], pq_jf_c[:3]
    a_q_g, b_q_g, c_q_g = pq_jf_a[3:], pq_jf_b[3:], pq_jf_c[3:]

    parent_q_inv = pq_jf_pa[3:] * np.array([-1., -1, -1, 1])

    target = c + c_delta

    # print("before", c)

    eps = 0.01
    lab = np.linalg.norm(b - a)
    lcb = np.linalg.norm(c - b)
    lat = np.clip(np.linalg.norm(target - a), eps, lab + lcb - eps)

    ac_ab_0 = np.arccos(np.clip(np.dot(normalize(c - a), normalize(b - a)), -1, 1))
    ba_bc_0 = np.arccos(np.clip(np.dot(normalize(a - b), normalize(c - b)), -1, 1))
    ac_at_0 = np.arccos(np.clip(np.dot(normalize(c - a), normalize(target - a)), -1, 1))

    ac_ab_1 = np.arccos(np.clip((lcb * lcb - lab * lab - lat * lat) / (-2 * lab * lat), -1, 1))
    ba_bc_1 = np.arccos(np.clip((lat * lat - lab * lab - lcb * lcb) / (-2 * lab * lcb), -1, 1))

    # print(ac_ab_0, ba_bc_0, ac_at_0, ac_ab_1, ba_bc_1)

    v = np.array([0., 0, -1]) if is_arm else np.array([0., 0, 1])  # else, legs, T pose elbow/knee pointing
    d = conversions.Q2R(a_q_g) @ v
    axis0_g = normalize(np.cross(c - a, d))
    # axis0_g = normalize(np.cross(c - a, b - a))
    axis1_g = normalize(np.cross(c - a, target - a))

    axis0_l = conversions.Q2R(parent_q_inv) @ axis0_g
    axis1_l = conversions.Q2R(a_q_g * np.array([-1., -1, -1, 1])) @ axis1_g

    r0 = conversions.A2Q(axis0_l * (ac_ab_1 - ac_ab_0))
    r1 = conversions.A2Q(axis0_l * (ba_bc_1 - ba_bc_0))
    r2 = conversions.A2Q(axis1_l * ac_at_0)

    a_q_l = quaternion.Q_mult(parent_q_inv, a_q_g)
    b_q_l = quaternion.Q_mult(a_q_g * np.array([-1., -1, -1, 1]), b_q_g)
    a_q_l_1 = quaternion.Q_mult(a_q_l, quaternion.Q_mult(r0, r2))
    # a_q_l_1 = quaternion.Q_mult(a_q_l, r0)
    b_q_l_1 = quaternion.Q_mult(b_q_l, r1)

    return a_q_l_1, b_q_l_1


def leg_two_joint_ik_keep_foot_pointing(
    pq_jf_pa, pq_jf_a, pq_jf_b, pq_jf_c, c_delta
) -> (np.ndarray, np.ndarray, np.ndarray):
    # when do two joint IK for leg, different from arm (we don't have hand modeled)
    # here we also rotate ankle so that foot global quat keep the same

    a_q_g, b_q_g, c_q_g = pq_jf_a[3:], pq_jf_b[3:], pq_jf_c[3:]
    pa_q_g = pq_jf_pa[3:]

    a_q_l_1, b_q_l_1 = two_joint_ik(pq_jf_pa, pq_jf_a, pq_jf_b, pq_jf_c, c_delta)

    a_q_g_1 = quaternion.Q_mult(pa_q_g, a_q_l_1)
    b_q_g_1 = quaternion.Q_mult(a_q_g_1, b_q_l_1)

    # c_q_g = q_mult(b_q_g, c_q_l)
    # c_q_l = q_mult(inv(b_q_g), c_q_g)
    # c_q_l_1 = q_mult(inv(b_q_g_1), c_q_g)
    c_q_l_1 = quaternion.Q_mult(b_q_g_1 * np.array([-1., -1, -1, 1]), c_q_g)

    return a_q_l_1, b_q_l_1, c_q_l_1
