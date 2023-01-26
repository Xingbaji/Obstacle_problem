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

device_num = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = device_num
device = torch.device('cuda:0')

config = {
    "path": '/home/lehu.sx/PDE_experiments/results/Obstacle_V3/',
    "name": 'model.pth',
    "num_linsapce": 101,
    "lr": 1e-3,
    "M": 500, # contact
    "N": 500, # boundary
    "num_iters": 10001,
    'visual': False,
    'depth': 4,
    'width':40,
    "loss_2_type": 'mean'
}
start_time = datetime.datetime.now().strftime('%m%d_%H%M')
config['path'] = os.path.join(config['path'],start_time)

class RitzNet(torch.nn.Module):
    def __init__(self, params):
        super(RitzNet, self).__init__()
        self.params = params
        self.linearIn = nn.Linear(self.params["dim_in"], self.params["width"])
        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.linear.append(nn.Linear(self.params["width"], self.params["width"]))

        self.linearOut = nn.Linear(self.params["width"], self.params["dim_out"])

    def forward(self, x):
        x = self.linearIn(x) # Match dimension
        x = F.relu(x**3)
        for layer in self.linear:
            x_temp = layer(x)
            x_temp = F.relu(x_temp)
            x = x_temp+x

        return self.linearOut(x)

#np part
M = config["M"]
N = config["N"]
num_iters = config["num_iters"]
num_linsapce = config["num_linsapce"]
X_axis = np.linspace(0,1,num_linsapce)
Y_axis = np.linspace(0,1,num_linsapce)
h = X_axis[1] - X_axis[0]
X,Y = np.meshgrid(X_axis,Y_axis)
f = -8
g = -0.3
h = 0

Input_boundary_x = np.concatenate([X[0,:],X[-1,:],X[:,0],X[:,-1]])
Input_boundary_y = np.concatenate([Y[0,:],Y[-1,:],Y[:,0],Y[:,-1]])
Input_boundary = np.stack([Input_boundary_x,Input_boundary_y])

Input_all_np = np.stack([X,Y],0)
Input_all_np = np.reshape(Input_all_np,[2,-1])

Input_inner = np.stack([X[1:-1,1:-1],Y[1:-1,1:-1]],0)
Input_inner = np.reshape(Input_inner,[2,-1])

if config['visual']==True:
    plt.plot(Input_inner[0,:],Input_inner[1,:],'r.');plt.show()
    plt.plot(Input_boundary[0,:],Input_boundary[1,:],'r.');plt.show()


Input_all = torch.from_numpy(Input_all_np.T).to(device)
Input_all.requires_grad = True
Input_inner = torch.from_numpy(Input_inner.T).to(device)
Input_boundary = torch.from_numpy(Input_boundary.T).to(device)

params={'dim_in' : 2, 'depth' : config['depth'],'width' : config['width'],'dim_out':1}
model = RitzNet(params)
model = model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[5000,8000,10000,15000], gamma=0.5)
for iter in range(num_iters):
    optimizer.zero_grad()
    U_all = model(Input_all) #[n.1]
    U_boundary = model(Input_boundary)
    nabla_u = torch.autograd.grad(U_all, Input_all, grad_outputs=torch.ones(U_all.size()).to(device), create_graph=True,retain_graph=True,only_inputs=True)[0] # [n,2]
    loss_1_flat = 1/2 * torch.sum(nabla_u**2,1,keepdim=True) - U_all*f #all part
    loss_1 = torch.mean(loss_1_flat)
    loss_2_flat = F.relu(g-U_all)**2 #contact part
    if config['loss_2_type'] == 'sum': #直接求和
        loss_2 = M * torch.sum(loss_2_flat)
    elif config['loss_2_type'] == 'mean': #求平均
        loss_2 = M * torch.mean(loss_2_flat)
    elif config['loss_2_type'] == 'mean_contact': #只对接触点进行平均
        num_nonzero = loss_2_flat.shape[0] - torch.sum(loss_2_flat==0)  # 如果直接平均会导致每个点的重要性一样，# tensor(2487) #这时候1/num_zero=0
        num_nonzero = float(num_nonzero)
        if num_nonzero == 0: # 避免除以0的情况
            num_nonzero = 1
        loss_2 = M * 1.0/num_nonzero * torch.sum(loss_2_flat)
    else:
        raise NotImplementedError
    loss_3_flat = (U_boundary- h)**2
    loss_3 = N * torch.mean(loss_3_flat)
    Loss = loss_1 + loss_2 + loss_3
    Loss.backward()
    optimizer.step()
    scheduler.step()

    if iter % 10 == 0:
        print('Train iter: [{}/{}] Loss: {:.6f} Loss_1: {:.10f}  Loss_2: {:.10f} Loss_3: {:.10f}'.format(
            iter,
            num_iters,
            Loss.item(),
            loss_1.item(),
            loss_2.item()/M,
            loss_3.item()/N)
        )
    if iter % 5000 == 0 and iter>100:
        fig = plt.figure()
        fig.set_size_inches(10, 10)
        fig.suptitle('iter'+str(iter)+'_'+config['loss_2_type']+'_M_'+str(config["M"])+'_N_'+str(config["N"]))
        U_np = U_all.cpu().detach().numpy()
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax = plt.axes(projection='3d')
        # ax.plot_trisurf(Input_all_np[0,:],Input_all_np[1,:],U_np,cmap='viridis', edgecolor='none');
        points = ax.scatter(Input_all_np[0,:],Input_all_np[1,:],U_np)
        ax.set_title('U')

        loss_1_flat_np = loss_1_flat.cpu().detach().numpy()
        ax = fig.add_subplot(2, 2, 2, projection='3d')
        points = ax.scatter(Input_all_np[0,:],Input_all_np[1,:],loss_1_flat_np)
        ax.set_title('loss_1')

        loss_2_flat_np = loss_2_flat.cpu().detach().numpy()
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        points = ax.scatter(Input_all_np[0,:],Input_all_np[1,:],loss_2_flat_np)
        ax.set_title('loss_2')

        loss_3_flat_np = loss_3_flat.cpu().detach().numpy()
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        points = ax.scatter(Input_boundary_x,Input_boundary_y,loss_3_flat_np)
        ax.set_title('loss_3')

        plt.show()
        print('')




if not os.path.exists(config['path']):
    os.makedirs(config['path'])
NAME = os.path.join(config['path'],config['name'])
torch.save(model.state_dict(), NAME)
with open(os.path.join(config['path'],'config.json'),'w') as handle:
    json.dump(config,handle,indent=4, sort_keys=False)

print(config)
