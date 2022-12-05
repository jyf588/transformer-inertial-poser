# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) Stanford University

import importlib.util
import pickle
import socket
import threading
import time
from datetime import datetime
import torch
import numpy as np
from fairmotion.ops import conversions
from pygame.time import Clock

from real_time_runner import RTRunner
from simple_transformer_with_state import TF_RNN_Past_State
from render_funcs import init_viz, update_height_field_pb, COLOR_OURS
# make deterministic
from learning_utils import set_seed
import constants as cst
set_seed(1234567)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
running = False
is_recording = True     # always record imu every 15 sec
record_buffer = None
num_imus = 6
num_float_one_frame = num_imus * 7      # sent from Xsens
FREQ = int(1. / cst.DT)

color = COLOR_OURS

model_name = "output/model-new-v0-2.pt"
USE_5_SBP = True
WITH_ACC_SUM = True
MULTI_SBP_CORRECTION = False
VIZ_H_MAP = True
MAX_ACC = 10.0

init_grid_np = np.random.uniform(-100.0, 100.0, (cst.GRID_NUM, cst.GRID_NUM))
init_grid_list = list(init_grid_np.flatten())

input_channels_imu = 6 * (9 + 3)
if USE_5_SBP:
    output_channels = 18 * 6 + 3 + 20
else:
    output_channels = 18 * 6 + 3 + 8

# make an aligned T pose, such that front is x, left is y, and up is z (i.e. without heading)
# the IMU sensor at head will be placed the same way, so we can get the T pose's heading (wrt ENU) easily
# the following are the known bone orientations at such a T pose
Rs_aligned_T_pose = np.array([
    1.0, 0, 0, 0, 0, -1, 0, 1, 0,
    1.0, 0, 0, 0, 0, -1, 0, 1, 0,
    1.0, 0, 0, 0, 0, -1, 0, 1, 0,
    1.0, 0, 0, 0, 0, -1, 0, 1, 0,
    1.0, 0, 0, 0, 0, -1, 0, 1, 0,
    1.0, 0, 0, 0, 0, -1, 0, 1, 0,
])
Rs_aligned_T_pose = Rs_aligned_T_pose.reshape((6, 3, 3))
Rs_aligned_T_pose = \
    np.einsum('ij,njk->nik', conversions.A2R(np.array([0, 0, np.pi/2])), Rs_aligned_T_pose)
print(Rs_aligned_T_pose)

# the state at the T pose, dq not necessary actually and will not be used either
s_init_T_pose = np.zeros(cst.n_dofs * 2)
s_init_T_pose[2] = 0.85
s_init_T_pose[3:6] = np.array([1.20919958, 1.20919958, 1.20919958])


# Based from TransPose github repo
class IMUSet:
    def __init__(self, imu_host='127.0.0.1', imu_port=27015):
        self.imu_host = imu_host
        self.imu_port = imu_port
        self.clock = Clock()

        self._imu_socket = None
        self._is_reading = False
        self._read_thread = None

        self.current_reading = None
        self.counter = 0

    def _read(self):
        """
        The thread that reads imu measurements into the buffer. It is a producer for the buffer.
        """
        data = ''
        while self._is_reading:
            data += self._imu_socket.recv(1024).decode('ascii')
            strs = data.split(' ', num_float_one_frame)

            # if we have read a whole frame
            if len(strs) == num_float_one_frame + 1:

                q_and_a_s = np.array(strs[:-1]).astype(float).reshape(num_imus, 7)

                q_s_gn = q_and_a_s[:, :4]
                R_s_gn = conversions.Q2R(q_s_gn)
                a_s = q_and_a_s[:, 4:]

                # need to do acc offset elsewhere.
                # a_s_g = np.einsum('ijk,ik->ij', R_s_g, a_s)
                # # probably doesn't matter, will be taken care by acc offset calibration as well.
                # a_s_g += np.array([0., 0., -9.8])

                # if self.counter % 25 == 0:
                #     print('\n' + str(q_s[0, :]) + str(a_s_g[0, :]))
                self.counter += 1
                # everything in global (ENU) frame
                self.current_reading = np.concatenate((R_s_gn.reshape(-1), a_s.reshape(-1)))

                data = strs[-1]         # if there are partial data from next frame
                self.clock.tick()

    def start_reading(self):
        """
        Start reading imu measurements into the buffer.
        """
        if self._read_thread is None:
            self._is_reading = True
            self._read_thread = threading.Thread(target=self._read)
            self._read_thread.setDaemon(True)
            self._imu_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._imu_socket.connect((self.imu_host, self.imu_port))
            self._read_thread.start()
        else:
            print('Failed to start reading thread: reading is already start.')

    def stop_reading(self):
        """
        Stop reading imu measurements.
        """
        if self._read_thread is not None:
            self._is_reading = False
            self._read_thread.join()
            self._read_thread = None
            self._imu_socket.close()


def get_input():
    global running
    while running:
        c = input()
        if c == 'q':
            running = False


def get_mean_readings_3_sec():
    counter = 0
    mean_buffer = []
    while counter <= FREQ * 3:
        clock.tick(FREQ)
        mean_buffer.append(imu_set.current_reading.copy())
        counter += 1

    return np.array(mean_buffer).mean(axis=0)


def get_transformed_current_reading():
    R_and_acc_t = imu_set.current_reading.copy()

    R_Gn_St = R_and_acc_t[: 6*9].reshape((6, 3, 3))
    acc_St = R_and_acc_t[6*9:].reshape((6, 3))

    R_Gp_St = np.einsum('nij,njk->nik', R_Gn_Gp.transpose((0, 2, 1)), R_Gn_St)
    R_Gp_Bt = np.einsum('nij,njk->nik', R_Gp_St, R_B0_S0.transpose((0, 2, 1)))

    acc_Gp = np.einsum('ijk,ik->ij', R_Gp_St, acc_St)
    acc_Gp = acc_Gp - acc_offset_Gp

    acc_Gp = np.clip(acc_Gp, -MAX_ACC, MAX_ACC)

    return np.concatenate((R_Gp_Bt.reshape(-1), acc_Gp.reshape(-1)))


def viz_point(x, ind):
    pb_c.resetBasePositionAndOrientation(
        p_vids[ind],
        x,
        [0., 0, 0, 1]
    )


if __name__ == '__main__':
    imu_set = IMUSet()

    ''' Load Character Info Moudle '''
    spec = importlib.util.spec_from_file_location(
        "char_info", "amass_char_info.py")
    char_info = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(char_info)

    pb_c, c1, _, p_vids, h_id, h_b_id = init_viz(char_info,
                                                 init_grid_list,
                                                 viz_h_map=VIZ_H_MAP,
                                                 hmap_scale=cst.GRID_SIZE,
                                                 gui=True,
                                                 compare_gt=False)

    model = TF_RNN_Past_State(
        input_channels_imu, output_channels,
        rnn_hid_size=512,
        tf_hid_size=1024, tf_in_dim=256,
        n_heads=16, tf_layers=4,
        dropout=0.0, in_dropout=0.0,
        past_state_dropout=0.8,
        with_acc_sum=WITH_ACC_SUM,
    )
    model.load_state_dict(torch.load(model_name))
    model = model.cuda()

    clock = Clock()
    imu_set.start_reading()

    input('Put all imus aligned with your body reference frame and then press any key.')
    print('Keep for 3 seconds ...', end='')

    # calibration: heading reset
    R_and_acc_mean = get_mean_readings_3_sec()

    # R_head = R_and_acc_mean[5*9: 6*9].reshape(3, 3)     # last sensor being head
    R_Gn_Gp = R_and_acc_mean[:6*9].reshape((6, 3, 3))
    # calibration: acceleration offset
    acc_offset_Gp = R_and_acc_mean[6*9:].reshape(6, 3)      # sensor frame (S) and room frame (Gp) align during this

    # R_head = np.array([[0.5,  0.866,  0.0],
    # [-0.866,  0.5,    0.0],
    # [ 0.0,  -0.0,  1.0]])

    # this should be pretty much just z rotation (i.e. only heading)
    # might be different for different sensors...
    print(R_Gn_Gp)

    input('\nWear all imus correctly and press any key.')
    for i in range(12, 0, -1):
        print('\rStand straight in T-pose and be ready. The calibration will begin after %d seconds.' % i, end='')
        time.sleep(1)
    print('\rStand straight in T-pose. Keep the pose for 3 seconds ...', end='')

    # calibration: bone-to-sensor transform
    R_and_acc_mean = get_mean_readings_3_sec()

    R_Gn_S0 = R_and_acc_mean[: 6 * 9].reshape((6, 3, 3))
    R_Gp_B0 = Rs_aligned_T_pose
    R_Gp_S0 = np.einsum('nij,njk->nik', R_Gn_Gp.transpose((0, 2, 1)), R_Gn_S0)
    R_B0_S0 = np.einsum('nij,njk->nik', R_Gp_B0.transpose((0, 2, 1)), R_Gp_S0)

    # # rotate init T pose according to heading reset results
    # nominal_root_R = conversions.A2R(s_init_T_pose[3:6])
    # root_R_init = R_head.dot(nominal_root_R)
    # s_init_T_pose[3:6] = conversions.R2A(root_R_init)

    # use real time runner with online data
    rt_runner = RTRunner(
        c1, model, 40, s_init_T_pose,
        map_bound=cst.MAP_BOUND,
        grid_size=cst.GRID_SIZE,
        play_back_gt=False,
        five_sbp=USE_5_SBP,
        with_acc_sum=WITH_ACC_SUM,
        multi_sbp_terrain_and_correction=MULTI_SBP_CORRECTION,
    )
    last_root_pos = s_init_T_pose[:3]     # assume always start from (0,0,0.9)

    print('\tFinish.\nStart estimating poses. Press q to quit')

    running = True

    get_input_thread = threading.Thread(target=get_input)
    get_input_thread.setDaemon(True)
    get_input_thread.start()

    RB_and_acc_t = get_transformed_current_reading()
    # rt_runner.record_raw_imu(RB_and_acc_t)
    if is_recording:
        record_buffer = RB_and_acc_t.reshape(1, -1)
    t = 1

    while running:
        RB_and_acc_t = get_transformed_current_reading()

        # t does not matter, not used
        res = rt_runner.step(RB_and_acc_t, last_root_pos, s_gt=None, c_gt=None, t=t)

        last_root_pos = res['qdq'][:3]

        viz_locs = res['viz_locs']
        for sbp_i in range(viz_locs.shape[0]):
            viz_point(viz_locs[sbp_i, :], sbp_i)

        if t % 15 == 0 and h_id is not None:
            # TODO: double for loop...
            for ii in range(init_grid_np.shape[0]):
                for jj in range(init_grid_np.shape[1]):
                    init_grid_list[jj * init_grid_np.shape[0] + ii] = \
                        rt_runner.region_height_list[rt_runner.height_region_map[ii, jj]]
            h_id, h_b_id = update_height_field_pb(
                pb_c,
                h_data=init_grid_list,
                scale=cst.GRID_SIZE,
                terrainShape=h_id,
                terrain=h_b_id
            )

        clock.tick(FREQ)

        # print('\r', R_G_Bt.reshape(6,9), acc_G_t, end='')

        t += 1
        # recording
        if is_recording:
            record_buffer = np.concatenate([record_buffer, RB_and_acc_t.reshape(1, -1)], axis=0)

            if t % (FREQ * 15) == 0:
                with open('../imu_recordings/r' + datetime.now().strftime('%m:%d:%T').replace(':', '-') + '.pkl',
                          "wb") as handle:
                    pickle.dump(
                        {"imu": record_buffer, "qdq_init": s_init_T_pose},
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL
                    )

    get_input_thread.join()
    imu_set.stop_reading()
    print('Finish.')
