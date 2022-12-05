# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) Stanford University

import numpy as np
import time
import random
import torch
from torch.utils.data import Dataset


class TrainSubDataset(Dataset):
    """
    Randomly Downsample the whole Training (mainly just AMASS) dataset (with down_sample_rates) to fit in memory.
    This is because each data point for Transformer is a time window which is too large if we don't downsample.
    After each epoch (downsampled data exhausted), you should get a new dataset instance
    therefore re-downsampling the dataset
    """

    def __init__(
             self,
             seq_length,
             info_path,
             imu_combine_path,
             s_combine_path,
             with_acc_sum=True,
            ):

        start_time = time.time()

        IMU_c = np.load(imu_combine_path)
        S_c = np.load(s_combine_path)
        infos = list(np.load(info_path))

        if with_acc_sum:
            IMU_sum_c = np.load(imu_combine_path.replace("imu", "sum_imu"))
        else:
            IMU_sum_c = None

        IMU = []
        S = []
        IMU_sum = []

        for info in infos:
            # each info is [start_t, end_t, down sample]
            start_t, end_t, down_sample_rate = tuple(info)
            time_range = range(start_t + seq_length, end_t - 1)

            if len(time_range) == 0:
                continue
            num_samples = np.maximum(round(len(time_range) / down_sample_rate), 1)

            # note, set random seed outside
            for t in random.sample(time_range, k=num_samples):
                # IMU dim (num_samples, T, num_feat)
                IMU.append(IMU_c[(t - seq_length):t, :])
                S.append(S_c[(t - seq_length):(t + 1), :])
                if with_acc_sum:
                    IMU_sum.append(IMU_sum_c[(t - seq_length):t, :])

        self.IMU = torch.from_numpy(np.array(IMU))
        self.IMU_sum = torch.from_numpy(np.array(IMU_sum))
        self.S = torch.from_numpy(np.array(S))
        self.size = (self.IMU.shape[0], self.IMU.shape[1])

        self.seq_length = seq_length
        self.with_acc_sum = with_acc_sum

        print("load time", time.time() - start_time)
        print("IMU shape", self.IMU.size())
        print("IMU sum shape", self.IMU_sum.size())
        print("S shape", self.S.size())

    def __getitem__(self, index):
        x_imu = self.IMU[index]

        if self.with_acc_sum:
            x_imu_acc_sum = self.IMU_sum[index]

        x_s = self.S[index, :-1, :]
        y_s_n = self.S[index, 1:, :]

        if self.with_acc_sum:
            x_imu = torch.cat((x_imu, x_imu_acc_sum), dim=1)

        return x_imu, x_s, y_s_n

    def __len__(self):
        return self.size[0]
