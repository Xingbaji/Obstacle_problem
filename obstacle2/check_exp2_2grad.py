""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import torch
import torch.utils.data as utils
import pdb
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import math
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torch
from torch.utils.data import DataLoader
import datetime
import pickle
import json
from scipy.integrate import solve_ivp
from matplotlib import cm
torch.set_default_tensor_type('torch.DoubleTensor')

device_num = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = device_num
device = torch.device('cuda:0')

config = {
    "path": '/home/lehu.sx/PDE_experiments/results/Obstacle_V3/',
    "name": 'model.pth',
    "num_linspace": 401,
    "lr": 1e-4,
    "alpha": 50, # project
    "beta": 50, # boundary
    "num_iters": 5001,
    'visual': False,
    'depth': 10,
    'width':100,
    "loss_2_type": 'mean'
}
start_time = datetime.datetime.now().strftime('%m%d_%H%M')
config['path'] = os.path.join(config['path'],start_time)

class NormalMultiLayersModel(nn.Module):
    def __init__(self,num_layers = 5, hidden_size = 50,output_size = 2,input_size = 1):
        """
        输入[x_i,x_j,y_i]输出[ep_ij]
        :param hidden_size:
        :param input_size:
        """
        super().__init__()
        h = hidden_size
        assert num_layers >=2
        self.fc = nn.ModuleList()
        self.ln = nn.ModuleList()
        self.fc.append(nn.Linear(input_size,h))
        self.ln.append(nn.LayerNorm(h))
        for _ in range(num_layers -2):
            self.fc.append(nn.Linear(h, h))
            self.ln.append(nn.LayerNorm(h))
        self.fc.append(nn.Linear(h, output_size))

    def forward(self, x):
        for i in range(len(self.fc)-1):
            # for layer in self.fc[:-1]:
            layer = self.fc[i]
            layernorm = self.ln[i]
            x = layer(x)
            x = layernorm(x)
            x = F.relu(x)**3
        x = self.fc[-1](x)
        return x

import sympy
x,y = sympy.symbols('x,y')
u = sympy.sqrt(1-sympy.sqrt(x**2+y**2))
ux = sympy.diff(u,x)
uy = sympy.diff(u,y)


#np part
alpha = config["alpha"]
beta = config["beta"]
f = 0
num_iters = config["num_iters"]

def generate_uniform(num_linsapce):
    X_axis = np.linspace(-2,2,num_linsapce)
    Y_axis = np.linspace(-2,2,num_linsapce)
    X,Y = np.meshgrid(X_axis,Y_axis)
    r = np.sqrt(X**2+Y**2)
    r_tmp = np.sqrt(1-r**2)
    cond = r<=1
    g = np.ones_like(r)*(-1)
    g[cond] = r_tmp[cond]
    r_star = 0.6979651482 #checked
    U_exact_all = np.zeros_like(r)
    cond1 = r<=r_star
    cond2 = r>r_star
    r_tmp2 = ((-1)*(r_star**2)*np.log(r/2))/np.sqrt(1-r_star**2)
    U_exact_all[cond1] = r_tmp[cond1]
    U_exact_all[cond2] = r_tmp2[cond2]

    if np.isnan(np.max(U_exact_all)) == True: #避免nan
        raise NotImplementedError

    if config['visual']==True:
        fig = plt.figure()
        # print('over')
    return X,Y,U_exact_all


#input_inner,input_boundary,U_exact_inner,U_exact_boundary,g_inner = generate_uniform(config['num_linspace'],config['num_linspace']*4)
X,Y,U = generate_uniform(config['num_linspace'])
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, U, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.close(fig)
delta = X[0,1]-X[0,0]
delta_u_xx = (U[2:,1:-1] - 2*U[1:-1,1:-1]+U[:-2,1:-1])/(delta**2)
delta_u_yy = (U[1:-1,2:] - 2*U[1:-1,1:-1]+U[1:-1,:-2])/(delta**2)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X[1:-1,1:-1], Y[1:-1,1:-1], delta_u_xx, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()
plt.close(fig)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X[1:-1,1:-1], Y[1:-1,1:-1], delta_u_yy, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
surf2 = ax.plot_surface(X, Y, U)
plt.show()
plt.close(fig)

fig = plt.figure()
ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X[150:-150,150:-150], Y[150:-150,150:-150], delta_u_xx[149:-149,149:-149], cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
surf2 = ax.plot_surface(X[150:-150,150:-150], Y[150:-150,150:-150], U[150:-150,150:-150])
plt.show()
plt.close(fig)
print('')

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X[1:-1,1:-1], Y[1:-1,1:-1], np.abs(delta_u_xx), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()
plt.close(fig)
