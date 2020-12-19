import pandas as pd
import torch
import torch.nn as nn

import numpy as np
import math


def read_csv(f):
    return pd.read_csv(f, header=None)


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

u_array = torch.from_numpy(u.values.astype(
    np.float32)).unsqueeze(0).unsqueeze(0)
v_array = torch.from_numpy(v.values.astype(
    np.float32)).unsqueeze(0).unsqueeze(0)
p_array = torch.from_numpy(p.values.astype(
    np.float32)).unsqueeze(0).unsqueeze(0)

# batch_c_y_x

# 連続の式
ddx = nn.Conv2d(1, 1, 3, 1, 0, bias=False)
ddx.weight.data = torch.tensor([[[[0, 0, 0],
                                  [-1., 1., 0],
                                  [0, 0, 0]]]])
ddy = nn.Conv2d(1, 1, 3, 1, 0, bias=False)
ddy.weight.data = torch.tensor([[[[0, -1., 0],
                                  [0, 1, 0],
                                  [0, 0, 0]]]])

dudx = ddx(u_array)/dx
dvdy = ddy(v_array)/dx

div_V = dudx + dvdy

print('div_v_max', torch.max(div_V))

print('div_V sum : ', math.sqrt(torch.sum(div_V ** 2, dim=(2, 3)).item())/(128*128))
