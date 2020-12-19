import numpy as np
import pickle
import pandas as pd
import glob
from multiprocessing.dummy import Pool
from tqdm import tqdm

csv_files_p = sorted(glob.glob("130_130_u0=1_5000step_Re100/*_p.csv"))
csv_files_u = sorted(glob.glob("130_130_u0=1_5000step_Re100/*_u.csv"))
csv_files_v = sorted(glob.glob("130_130_u0=1_5000step_Re100/*_v.csv"))

# csv_files_p = sorted(glob.glob("input_test/*_p.csv"))
# csv_files_u = sorted(glob.glob("input_test/*_u.csv"))
# csv_files_v = sorted(glob.glob("input_test/*_v.csv"))
print('finish glob')

def read_csv(f):
    return pd.read_csv(f, header=None)

with Pool() as p:
    arr_p = p.map(read_csv, csv_files_p)

with Pool() as p:
    arr_u = p.map(read_csv, csv_files_u)

with Pool() as p:
    arr_v = p.map(read_csv, csv_files_v)

#print(arr_p)

print("finish read_csv")

dx = np.array([1.0/128])
dy = np.array([1.0/128])
Re = np.array([100.0])
u0 = np.array([10.0])
cn = np.array([0.5])
nend = np.array([500])
visc = np.array([10.0/Re])
dt = np.array([cn/(u0/dx + 2*visc*(1/dx/dx + 1/dy/dy))][0][0])

data_dict = {'dt':dt, 'dx':dx, 'dy':dy, 'Re':Re, 'u0':u0, 'nend':nend}

def extract_time(filename):
    return round(float(filename[-14:-6]) - 0.000064, 5)

time_list = []

for filename in tqdm(csv_files_p):
    time = extract_time(filename)
    time_list.append(time)

# for p in arr_p:
#     print('??????????????????')
#     print(p)
#     print(type(p))
#     print(p.values)
#     print(type(p.values))

p_array = np.stack([p.values for p in tqdm(arr_p)])
u_array = np.stack([u.values for u in tqdm(arr_u)])
v_array = np.stack([v.values for v in tqdm(arr_v)])

from pprint import pprint
# pprint(data_dict)

data_dict['t'] = np.array(time_list)
data_dict['p'] = p_array
data_dict['u'] = u_array
data_dict['v'] = v_array
data_dict['x'] = np.linspace(0, 1.0, num=128, endpoint=False)
data_dict['y'] = np.linspace(0, 1.0, num=128, endpoint=False)

pprint(data_dict)

with open('1216_130_130_5000step.pickle', 'wb') as f:
    pickle.dump(data_dict, f)