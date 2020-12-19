import pandas as pd
import torch
import torch.nn as nn

import numpy as np
import math

mse_loss = nn.MSELoss()


def read_csv(f):
    return pd.read_csv(f, header=None)


def compare_tensor(a, b):
    a = a.contiguous().view(-1)
    b = b.contiguous().view(-1)
    return mse_loss(a, b).item()


u_filename = '/home/takeshi/GAN/PI_SRGAN/PI_loss_test_dir/130_130_stationary/13.567198_u.csv'
v_filename = '/home/takeshi/GAN/PI_SRGAN/PI_loss_test_dir/130_130_stationary/13.567198_v.csv'
p_filename = '/home/takeshi/GAN/PI_SRGAN/PI_loss_test_dir/130_130_stationary/13.567198_p.csv'

with open(u_filename) as f:
    u = read_csv(f)

with open(v_filename) as f:
    v = read_csv(f)

with open(p_filename) as f:
    p = read_csv(f)

dx = torch.tensor([1./128])
ni = 128
nj = 128

u_array = torch.from_numpy(u.values.astype(
    np.float32)).unsqueeze(0).unsqueeze(0)
v_array = torch.from_numpy(v.values.astype(
    np.float32)).unsqueeze(0).unsqueeze(0)
p_array = torch.from_numpy(p.values.astype(
    np.float32)).unsqueeze(0).unsqueeze(0)

visc = 1. / 100

visc_per_dx = visc / dx

ue_for = torch.zeros(1, 1, nj + 1, ni + 1)
vn_for = torch.zeros(1, 1, nj + 1, ni + 1)

flux_e_for = torch.zeros(1, 1, nj + 1, ni + 1)
flux_n_for = torch.zeros(1, 1, nj + 1, ni + 1)

for i in range(ni + 1):
    for j in range(nj + 1):
        ue_for[:, :, j, i] = (u_array[:, :, j, i] +
                              u_array[:, :, j + 1, i]) / 2
        vn_for[:, :, j, i] = (v_array[:, :, j, i] +
                              v_array[:, :, j + 1, i]) / 2

        flux_e_for[:, :, j, i] = ue_for[:, :, j, i] * (v_array[:, :, j, i + 1] + v_array[:, :, j, i]) / 2 - \
            visc_per_dx * (v_array[:, :, j, i + 1] - v_array[:, :, j, i])

        flux_n_for[:, :, j, i] = vn_for[:, :, j, i] * (v_array[:, :, j + 1, i] + v_array[:, :, j, i]) / 2 - \
            visc_per_dx * (v_array[:, :, j + 1, i] - v_array[:, :, j, i])

dvdt_max_for = 0.0

e_diff_for = torch.zeros(1, 1, nj, ni + 1)
n_diff_for = torch.zeros(1, 1, nj, ni + 1)
p_diff_for = torch.zeros(1, 1, nj, ni + 1)

for i in range(1, ni + 1):
    for j in range(1, nj):
        a = -(-flux_e_for[:, :, j, i - 1] + flux_e_for[:, :, j, i]) / dx
        b = -(-flux_n_for[:, :, j - 1, i] + flux_n_for[:, :, j, i]) / dx
        c = -(-p_array[:, :, j, i] + p_array[:, :, j + 1, i]) / dx
        e_diff_for[:, :, j, i] = a
        n_diff_for[:, :, j, i] = b
        p_diff_for[:, :, j, i] = c

        dvdt = a + b + c
        if dvdt > dvdt_max_for:
            dvdt_max_for = dvdt

ave_x = nn.Conv2d(1, 1, 2, 1, 0, bias=False)
ave_x.weight.data = torch.tensor([[[[0.5, 0.5],
                                    [0., 0.]]]])

ave_y = nn.Conv2d(1, 1, 2, 1, 0, bias=False)
ave_y.weight.data = torch.tensor([[[[0.5, 0.],
                                    [0.5, 0.]]]])

ddx_up = nn.Conv2d(1, 1, 2, 1, 0, bias=False)
ddx_up.weight.data = torch.tensor([[[[-1., 1.],
                                     [0, 0]]]])

ddy_up = nn.Conv2d(1, 1, 2, 1, 0, bias=False)
ddy_up.weight.data = torch.tensor([[[[-1., 0],
                                     [1., 0]]]])

ue_conv = ave_y(u_array)
vn_conv = ave_y(v_array)

ue_diff = compare_tensor(ue_conv, ue_for)
vn_diff = compare_tensor(vn_conv, vn_for)

flux_e_conv = ue_conv * ave_x(v_array) - visc_per_dx * ddx_up(v_array)
flux_n_conv = vn_conv * ave_y(v_array) - visc_per_dx * ddy_up(v_array)

flux_e_diff = compare_tensor(flux_e_conv, flux_e_for)
flux_n_diff = compare_tensor(flux_n_conv, flux_n_for)

flux_e_conv = flux_e_conv[:, :, :-1, :]
flux_n_conv = flux_n_conv[:, :, :-1, :]
p_cut = p_array[:, :, 1:-1, :-1]

ddx_down = nn.Conv2d(1, 1, 2, 1, 0, bias=False)
ddx_down.weight.data = torch.tensor([[[[0., 0],
                                       [-1., 1.]]]])

ddy_down = nn.Conv2d(1, 1, 2, 1, 0, bias=False)
ddy_down.weight.data = torch.tensor([[[[0, -1.],
                                       [0, 1.]]]])

e_diff_conv = -ddx_down(flux_e_conv) / dx
n_diff_conv = -ddy_down(flux_n_conv) / dx
p_diff_conv = -ddy_down(p_cut) / dx

e_diff_diff = compare_tensor(e_diff_conv, e_diff_for[:, :, 1:, 1:])
n_diff_diff = compare_tensor(n_diff_conv, n_diff_for[:, :, 1:, 1:])
p_diff = compare_tensor(p_diff_conv, p_diff_for[:, :, 1:, 1:])

dudt = e_diff_conv + n_diff_conv + p_diff_conv
dudt_max_conv = torch.max(dudt)

print()
