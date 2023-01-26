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

device_num = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = device_num
device = torch.device('cuda:0')

config = {
    "path": '/home/sx/PDE_experiments/results/Obstacle_V3/',
    "name": 'model.pth',
    "num_linsapce": 100,
    "lr": 1e-4,
    "alpha": 50, # project
    "beta": 50, # boundary
    "num_iters": 8001,
    'visual': False,
    'depth': 10,
    'width':100,
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
        x = F.relu(x)
        for layer in self.linear:
            x_temp = layer(x)
            x_temp = F.relu(x_temp)
            x = x_temp+x

        return self.linearOut(x)

#np part
alpha = config["alpha"]
beta = config["beta"]
f = 0
num_iters = config["num_iters"]
num_linsapce = config["num_linsapce"]
X_axis = np.linspace(-2,2,num_linsapce)
Y_axis = np.linspace(-2,2,num_linsapce)
X,Y = np.meshgrid(X_axis,Y_axis)
X_flat = np.reshape(X,[-1,1])
Y_flat = np.reshape(Y,[-1,1])

Input_boundary_x = np.concatenate([X[0,:],X[-1,:],X[:,0],X[:,-1]])
Input_boundary_y = np.concatenate([Y[0,:],Y[-1,:],Y[:,0],Y[:,-1]])
Input_boundary = np.stack([Input_boundary_x,Input_boundary_y],1)

Input_all_np = np.stack([X_flat,Y_flat],1).squeeze()
# Input_inner = np.stack([X[1:-1,1:-1],Y[1:-1,1:-1]],0)
# Input_inner = np.reshape(Input_inner,[-1,2])

if config['visual']==True:
    # plt.plot(Input_inner[:,0],Input_inner[:,1],'r.');plt.title('inner');plt.show()
    plt.plot(Input_boundary[:,0],Input_boundary[:,1],'r.');plt.title('boundary');plt.show()

#compute g
r = np.sqrt(X_flat**2+Y_flat**2)
r_tmp = 1-X_flat**2-Y_flat**2
cond = r<=1
g = np.ones_like(X_flat)*(-1)
g[cond] = np.sqrt(r_tmp[cond]+1e-15)
g_np = np.reshape(g,[-1,1])
if config['visual']==True:
    # check g image
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title('g')
    points = ax.scatter(X, Y, g)
    fig.tight_layout()
    fig.show()

#Exact solution on all parts
r_star = 0.6979651482 #checked
U_exact = np.zeros_like(X_flat)
cond1 = r<=r_star
cond2 = r>r_star
r_tmp2 = ((-1)*(r_star**2)*np.log(r/2))/np.sqrt(1-r_star**2)
U_exact[cond1] = np.sqrt(r_tmp[cond1]+1e-15)
U_exact[cond2] = r_tmp2[cond2]
U_exact = np.reshape(U_exact,[-1,1])
U_exact_torch =  torch.from_numpy(U_exact).to(device)
if config['visual']==True:
    # check psi image
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title('U_exact')
    points = ax.scatter(X, Y, U_exact)
    # surf = ax.plot_surface(X,Y,U_exact)
    fig.tight_layout()
    fig.show()

# boundary part
r_boundary = np.sqrt(Input_boundary_x**2+Input_boundary_y**2)
U_boundary = np.zeros_like(Input_boundary_x)
cond1 = r_boundary<=r_star
cond2 = r_boundary>r_star
r_tmp = 1-r_boundary**2
r_tmp2 = ((-1)*(r_star**2)*np.log(r_boundary/2))/np.sqrt(1-r_star**2)
U_boundary[cond1] = np.sqrt(r_tmp[cond1]+1e-15)
U_boundary[cond2] = r_tmp2[cond2]
U_boundary = np.reshape(U_boundary,[-1,1])
U_boundary_torch =  torch.from_numpy(U_boundary).to(device)
h = U_boundary_torch
if config['visual']==True:
    # check psi image
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title('U_bounadry')
    points = ax.scatter(Input_boundary_x, Input_boundary_y, U_boundary)
    # surf = ax.plot_surface(X,Y,U_exact)
    fig.tight_layout()
    fig.show()


Input_all = torch.from_numpy(Input_all_np).to(device)
Input_all.requires_grad = True
# Input_inner = torch.from_numpy(Input_inner.T).to(device)
Input_boundary = torch.from_numpy(Input_boundary).to(device)
g_torch = torch.from_numpy(g_np).to(device)

params={'dim_in' : 2, 'depth' : config['depth'],'width' : config['width'],'dim_out':1}
model = RitzNet(params)
model = model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[4000,8000,15000], gamma=0.1)
for iter in range(num_iters):
    optimizer.zero_grad()
    U_all = model(Input_all) #[n.1]
    U_boundary_pred = model(Input_boundary)
    nabla_u = torch.autograd.grad(U_all, Input_all, grad_outputs=torch.ones(U_all.size()).to(device), create_graph=True,retain_graph=True,only_inputs=True)[0] # [n,2]
    loss_1_flat = 1/2 * torch.sum(nabla_u**2,1,keepdim=True) - U_all*f #all part
    loss_1 = torch.mean(loss_1_flat)
    loss_2_flat = F.relu(g_torch-U_all)**2 #contact part
    if config['loss_2_type'] == 'sum': #直接求和
        loss_2 = alpha * torch.sum(loss_2_flat)
    elif config['loss_2_type'] == 'mean': #求平均
        loss_2 = alpha * torch.mean(loss_2_flat)
    elif config['loss_2_type'] == 'mean_contact': #只对接触点进行平均
        num_nonzero = loss_2_flat.shape[0] - torch.sum(loss_2_flat==0)  # 如果直接平均会导致每个点的重要性一样，# tensor(2487) #这时候1/num_zero=0
        num_nonzero = float(num_nonzero)
        if num_nonzero == 0: # 避免除以0的情况
            num_nonzero = 1
        loss_2 = alpha * 1.0/num_nonzero * torch.sum(loss_2_flat)
    else:
        raise NotImplementedError
    loss_3_flat = (U_boundary_pred- h)**2
    loss_3 = beta * torch.mean(loss_3_flat)
    Loss = loss_1 + loss_2 + loss_3
    Loss.backward()
    optimizer.step()
    # scheduler.step()
    loss_exact = torch.mean(torch.abs(U_exact_torch - U_all))

    if iter % 10 == 0:
        print('Train iter: [{}/{}] Loss: {:.6f} Loss_1: {:.10f}  Loss_2: {:.6f} Loss_3: {:.6f},Loss_exact: {:.6f}'.format(
            iter,
            num_iters,
            Loss.item(),
            loss_1.item(),
            loss_2.item()/alpha,
            loss_3.item()/beta,
            loss_exact.item())
        )
    if iter % 2000 == 0 and iter>1:
        fig = plt.figure()
        fig.set_size_inches(10, 10)
        fig.suptitle('iter'+str(iter)+'_'+config['loss_2_type']+'_alpha_'+str(config["alpha"])+'_beta_'+str(config["beta"]))
        U_np = U_all.cpu().detach().numpy()
        ax = fig.add_subplot(3, 2, 1, projection='3d')
        points = ax.scatter(Input_all_np[:,0],Input_all_np[:,1],U_np)
        ax.set_title('U_pred')

        U_np = U_all.cpu().detach().numpy()
        ax = fig.add_subplot(3, 2, 2, projection='3d')
        points = ax.scatter(Input_all_np[:,0],Input_all_np[:,1],U_np - U_exact)
        ax.set_title('U_bias')

        loss_1_flat_np = loss_1_flat.cpu().detach().numpy()
        ax = fig.add_subplot(3, 2, 3, projection='3d')
        points = ax.scatter(Input_all_np[:,0],Input_all_np[:,1],loss_1_flat_np)
        ax.set_title('loss_1')

        loss_2_flat_np = loss_2_flat.cpu().detach().numpy()
        ax = fig.add_subplot(3, 2, 4, projection='3d')
        points = ax.scatter(Input_all_np[:,0],Input_all_np[:,1],loss_2_flat_np)
        ax.set_title('loss_2')

        loss_3_flat_np = loss_3_flat.cpu().detach().numpy()
        ax = fig.add_subplot(3, 2, 5, projection='3d')
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
