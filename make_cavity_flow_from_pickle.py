import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.image import imread

# fig, ax = plt.subplots()
# plt.colorbar()

images_u = []

with open('data/cavity_flow_128_128/1201_data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

p_array = data_dict['p']
u_array = data_dict['u']
v_array = data_dict['v']

nend = data_dict['nend'][0]

# def plot(i):
#     print(i)
#     plt.cla()
#     raw_image = p_array[:, :, 50 * i]
#     im = ax.imshow(raw_image[:, :].transpose(), vmin=-0.5, vmax=0.5)
#     if i == 0:
#         fig.colorbar(im)
# n = n + 1

np.save('data/cavity_flow_128_128/HR_128_128/0_4999_b_c(u_v_p)_w_h.npy', np.stack([np.stack([u_array[:, :, n].transpose(), v_array[:, :,
                                                                                                                                   n].transpose(), p_array[:, :, n].transpose()], axis=2).transpose(2, 0, 1) for n in tqdm(range(5000))], axis=3).transpose(3, 0, 1, 2))
np.save('data/cavity_flow_128_128/HR_128_128/5000_9999_b_c(u_v_p)_w_h.npy', np.stack([np.stack([u_array[:, :, n].transpose(), v_array[:, :,
                                                                                                                                      n].transpose(), p_array[:, :, n].transpose()], axis=2).transpose(2, 0, 1) for n in tqdm(range(5000, 10000))], axis=3).transpose(3, 0, 1, 2))

# for n in tqdm(range(nend)):
#     im = np.stack([u_array[:, :, n].transpose(), v_array[:, :,
#                                                          n].transpose(), p_array[:, :, n].transpose()], axis=2).transpose(2, 0, 1)
# print(im.shape)
#cv2.imwrite('data/cavity_flow_128_128/HR_128_128/{}.jpg'.format(n), im)
# np.save('data/cavity_flow_128_128/HR_128_128/{}.npy'.format(n), im)
# ani = animation.FuncAnimation(fig, plot, interval=1)
# plt.show()

# for n in tqdm(range(nend)):
#     p = p_array[:, :, n]
#     u = u_array[:, :, n]
#     v = v_array[:, :, n]
#     break
