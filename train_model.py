# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) Stanford University

import argparse
import os
import sys
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from simple_transformer_with_state import TF_RNN_Past_State
from training_data_loader import TrainSubDataset
from learning_utils import set_seed, loss_q_only_2axis, loss_constr_multi, loss_jerk

sys.path.append("../")
torch.set_printoptions(threshold=10_000)

parser = argparse.ArgumentParser(description='Transformer Training for IMU')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size (default: 128)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: False)')
parser.add_argument('--rnn_dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--in_dropout', type=float, default=0.0,
                    help='dropout applied to IMU input (default: 0.0)')
parser.add_argument('--clip', type=float, default=5.0,
                    help='gradient clip, -1 means no clip (default: 5.0)')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit (default: 10)')
parser.add_argument('--seq_len', type=int, default=40,
                    help='sequence window length for input (default: 40)')
parser.add_argument('--log-interval', type=int, default=100,
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=4e-4,
                    help='initial learning rate (default: 4e-4)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--weight_decay', type=float, default='1e-5',
                    help='for AdamW')
parser.add_argument('--rnn_nhid', type=int, default=512,
                    help='hidden size of rnn (default: 512)')
parser.add_argument('--tf_nhid', type=int, default=1024,
                    help='hidden size of transformer')
parser.add_argument('--tf_in_dim', type=int, default=256,
                    help='input dimension of transformer')
parser.add_argument('--n_heads', type=int, default=8,
                    help='num of heads for transformer')
parser.add_argument('--tf_layers', type=int, default=4,
                    help='num of layers for transformer')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--save_path', type=str, default='output/model-tmp',
                    help='model save path')
parser.add_argument('--cosine_lr', action='store_true',
                    help='use cosine learning rate (default: False)')
parser.add_argument('--warm_start', type=str, default=None,
                    help='')
parser.add_argument('--double', action='store_true',
                    help='use double precision instead of single')
parser.add_argument('--past_dropout', type=float, default=0.8,
                    help='input dropout for past state in transformer')
parser.add_argument('--with_acc_sum', action='store_true',
                    help='')
parser.add_argument('--n_sbps', type=int, default=5,
                    help='')
parser.add_argument('--noise_input_hist', type=float, default=0.1,
                    help='')
parser.add_argument('--data_version_tag', type=str, default=None,
                    help='')
args = parser.parse_args()

batch_size = args.batch_size
seq_length = args.seq_len
epochs = args.epochs
n_sbps = args.n_sbps
with_acc_sum = args.with_acc_sum
d_tag = args.data_version_tag
noise_input_hist = args.noise_input_hist

if args.double:
    torch.set_default_dtype(torch.float64)

set_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print(args)
print("Preparing data...")

input_channels = 6 * (9 + 3)
output_channels = 18 * 6 + 3 + (n_sbps * 4)

model = TF_RNN_Past_State(
    input_channels, output_channels,
    rnn_hid_size=args.rnn_nhid,
    tf_hid_size=args.tf_nhid, tf_in_dim=args.tf_in_dim,
    n_heads=args.n_heads, tf_layers=args.tf_layers,
    dropout=args.rnn_dropout, in_dropout=args.in_dropout,
    past_state_dropout=args.past_dropout,
    with_rnn=True,
    with_acc_sum=with_acc_sum
)

if args.warm_start is not None:
    model.load_state_dict(torch.load(args.warm_start + ".pt"))
    # TODO: better also to load Adam state

if args.cuda:
    model.cuda()

lr = args.lr

if args.optim == "AdamW":
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, weight_decay=args.weight_decay)
else:
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

if args.cosine_lr:
    lr_s = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs + 850)      # 850 probably doesn't matter
else:
    lr_s = None


def train(epoch):
    # torch.autograd.set_detect_anomaly(True)

    model.train()

    data = TrainSubDataset(
        seq_length=seq_length,
        imu_combine_path="data/imu_train_" + d_tag + ".npy",
        s_combine_path="data/s_train_" + d_tag + ".npy",
        info_path="data/info_train_" + d_tag + ".npy",
        with_acc_sum=with_acc_sum,
    )

    num_samples = len(data)

    loader = DataLoader(data, shuffle=True, pin_memory=True,
                        batch_size=batch_size,
                        num_workers=1)

    batch_idx = 1
    total_loss = 0
    i = 0

    for (x_imu, x_s, y) in loader:

        i += x_imu.size()[0]

        start = time.time()

        loss_func = loss_q_only_2axis
        loss_func_c = loss_constr_multi

        if args.double:
            x_imu = x_imu.double()
            x_s = x_s.double()
            y = y.double()
        if args.cuda:
            x_imu = x_imu.cuda()
            x_s = x_s.cuda()
            y = y.cuda()

        # TODO: not sure what's the best value for this parameter
        noise_s = (torch.rand(x_s.size()) - 0.5) * (noise_input_hist * 2)
        if args.cuda:
            noise_s = noise_s.cuda()

        y_pred = model(x_imu, x_s + noise_s)

        loss_j = loss_jerk(y_pred[:, :, :-3-(n_sbps * 4)])

        y_pred = y_pred.reshape(-1, y_pred.size()[-1])
        y = y.reshape(-1, y.size()[-1])

        loss_q = loss_func(y[:, :-(n_sbps * 4)], y_pred[:, :-(n_sbps * 4)])
        loss_c = loss_func_c(y[:, -(n_sbps * 4):], y_pred[:, -(n_sbps * 4):])

        loss = loss_c + loss_q
        if loss_j is not None:
            loss += loss_j

        total_loss += loss.item()
        optimizer.zero_grad()

        loss.backward()

        total_norm = None
        if args.clip > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        batch_idx += 1

        if args.cosine_lr:
            lr_s.step()
            cur_lr = lr_s.get_last_lr()[0]
        else:
            cur_lr = lr

        end = time.time()
        if batch_idx % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = min(i, num_samples)
            # for logging
            print("total norm", total_norm)
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.7f}\tLoss: {:.6f}\tEp Time: {:.4f}'
                  .format(epoch, processed, num_samples, 100. * processed / num_samples, cur_lr, cur_loss, end - start),
                  flush=True)
            total_loss = 0


def save(m, ep_num):
    if ep_num == 1 or ep_num % 10 == 0:
        save_filename = os.path.join(args.save_path, "it" + str(ep_num) + ".pt")
        torch.save(m.state_dict(), save_filename)
        print('Saved as %s' % save_filename)
    torch.save(m.state_dict(), args.save_path + ".pt")


def evaluate(ep_num):
    model.eval()
    print("Saving...")
    save(model, ep_num)
    return


try:
    os.makedirs(args.save_path)
except FileExistsError:
    print("warning: path existed")
except OSError:
    exit()

for ep in range(1, epochs + 1):
    evaluate(ep)
    train(ep)
