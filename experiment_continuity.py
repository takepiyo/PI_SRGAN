import pickle
from pprint import pprint

# with open('data/1214_data.pickle', 'rb') as f:
#     data_dict = pickle.load(f)

with open('data/1215_smalldata.pickle', 'rb') as f:
    data_dict = pickle.load(f)

u = data_dict['u']
v = data_dict['v']
p = data_dict['p']

print()

u_1 = u[1]
v_1 = v[1]

print()
