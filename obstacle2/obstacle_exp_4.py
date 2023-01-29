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
    "num_linspace": 201,
    "lr": 1e-4,
    "alpha": 50, # project
    "beta": 50, # boundary
    "num_iters": 1501,
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


def g_fun(input_xy):
    x = input_xy[:,0]
    y = input_xy[:,1]
    G = np.zeros_like(x)
    cond1 = np.abs(x - 0.6) + np.abs(y - 0.6) < 0.04
    cond2 = (x-0.6)**2 + (y-0.25)**2 <0.001
    cond3 = (x<0.13) * (x>0.075) * (y==0.57)
    G[cond1] = 5.0
    G[cond2] = 4.5
    G[cond3] = 4.5
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x,y,G,cmap='viridis', edgecolor='none');plt.show()
    return G





#np part
alpha = config["alpha"]
beta = config["beta"]
f = 0
num_iters = config["num_iters"]

def generate_uniform(num_linsapce):
    X_axis = np.linspace(-1,1,num_linsapce)
    Y_axis = np.linspace(-1,1,num_linsapce)
    X,Y = np.meshgrid(X_axis,Y_axis)
    X_flat = np.reshape(X,[-1,1]).squeeze()
    Y_flat = np.reshape(Y,[-1,1]).squeeze()
    input_uniform = np.stack([X_flat,Y_flat],1).squeeze()

    input_boundary_x = np.concatenate([X[0,:],X[-1,:],X[:,0],X[:,-1]])
    input_boundary_y = np.concatenate([Y[0,:],Y[-1,:],Y[:,0],Y[:,-1]])
    input_boundary = np.stack([input_boundary_x,input_boundary_y],1)
    num_boundary = input_boundary.shape[0]
    input_all = np.concatenate([input_boundary,input_uniform])
    g = g_fun(input_all)



    U_exact_boundary = U_exact_all[:num_boundary]
    U_exact_uniform = U_exact_all[num_boundary:]

    if np.isnan(np.max(U_exact_all)) == True: #避免nan
        raise NotImplementedError

    if config['visual']==True:
        fig = plt.figure()
        plt.plot(input_boundary[:,0],input_boundary[:,1],'x')
        plt.plot(input_uniform[:,0],input_uniform[:,1],'.')
        plt.title('sample points')
        fig.show()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title('g')
        ax.plot_trisurf(input_uniform[:,0],input_uniform[:,1], g,cmap='viridis', edgecolor='none');
        fig.show()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title('U')
        # points = ax.scatter(input_all[:,0],input_all[:,1], U_exact,c=U_exact, cmap='viridis', linewidth=0.5)
        ax.plot_trisurf(input_uniform[:,0],input_uniform[:,1], U_exact_uniform,cmap='viridis', edgecolor='none');
        fig.show()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title('U')
        # points = ax.scatter(input_all[:,0],input_all[:,1], U_exact,c=U_exact, cmap='viridis', linewidth=0.5)
        ax.scatter(input_boundary[:,0],input_boundary[:,1], U_exact_boundary);
        fig.show()

    U_exact_uniform = np.reshape(U_exact_uniform,[-1,1])
    U_exact_boundary = np.reshape(U_exact_boundary,[-1,1])
    g = np.reshape(g,[-1,1])

    # print('over')
    return input_uniform,input_boundary,U_exact_uniform,U_exact_boundary,g


#input_inner,input_boundary,U_exact_inner,U_exact_boundary,g_inner = generate_uniform(config['num_linspace'],config['num_linspace']*4)
input_inner,input_boundary,U_exact_inner,U_exact_boundary,g_inner = generate_uniform(config['num_linspace'])
input_inner_torch = torch.from_numpy(input_inner).to(device)
input_inner_torch.requires_grad = True
input_boundary_torch = torch.from_numpy(input_boundary).to(device)
U_exact_boundary_torch = torch.from_numpy(U_exact_boundary).to(device)
U_exact_inner_torch = torch.from_numpy(U_exact_inner).to(device)
g_torch = torch.from_numpy(g_inner).to(device)

model = NormalMultiLayersModel(num_layers=config['depth'],hidden_size=config['width'],output_size = 1,input_size = 2)
model = model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[1200,5000.10000], gamma=0.1)

Log = {'iter':[],'loss':[],'loss_1':[],'loss_2':[],'loss_3':[],'loss_exact':[]}

iter = 0
while iter <= num_iters:
    optimizer.zero_grad()
    U_inner_pred = model(input_inner_torch) #[n.1]
    # U_boundary_pred = U_all[:num_boundary]
    U_boundary_pred  = model(input_boundary_torch)
    nabla_u = torch.autograd.grad(U_inner_pred, input_inner_torch, grad_outputs=torch.ones(U_inner_pred.size()).to(device), create_graph=True,retain_graph=True,only_inputs=True)[0] # [n,2]
    loss_1_flat = 1/2 * torch.sum(nabla_u**2,1,keepdim=True) - U_inner_pred*f #all part
    loss_1 = torch.mean(loss_1_flat)
    Topk, index = torch.topk(torch.abs(loss_1_flat.squeeze()),10)
    loss_1_top = torch.mean(Topk)

    loss_2_flat = F.relu(g_torch-U_inner_pred)**2 #contact part
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
    loss_3_flat = (U_boundary_pred-U_exact_boundary_torch)**2
    loss_3 = beta * torch.mean(loss_3_flat)
    Loss = loss_1 + loss_2 + loss_3
    Loss.backward()
    optimizer.step()
    scheduler.step()
    loss_exact = torch.mean(torch.abs(U_inner_pred - U_exact_inner_torch))
    loss_exact_l2 = torch.mean((U_inner_pred - U_exact_inner_torch)**2)

    if loss_exact <1e-2:
        if not os.path.exists(config['path']):
            os.makedirs(config['path'])
        NAME = os.path.join(config['path'],config['name'])
        torch.save(model.state_dict(), NAME)
        with open(os.path.join(config['path'],'config.json'),'w') as handle:
            json.dump(config,handle,indent=4, sort_keys=False)

        print(loss_exact,loss_exact_l2,config)
        break

    if iter % 10 == 0:
        print('Train iter: [{}/{}] Loss: {:.6f} Loss_1: {:.10f} Loss_1_top: {:.10f}   Loss_2: {:.10f} Loss_3: {:.10f},Loss_exact: {:.6f},Loss_exact_l2: {:.6f}'.format(
            iter,
            num_iters,
            Loss.item(),
            loss_1.item(),
            loss_1_top.item(),
            loss_2.item()/alpha,
            loss_3.item()/beta,
            loss_exact.item(),
            loss_exact_l2.item(),
            ))

        if iter == 1200:
            alpha = alpha * 10
            # beta = beta * 10
            print('alpha: ',alpha,' beta: ',beta)
        # #

        if iter % 1000 == 0 and iter >900:
            fig = plt.figure()
            fig.set_size_inches(10, 10)
            fig.suptitle('iter'+str(iter)+'_'+config['loss_2_type']+'_alpha_'+str(config["alpha"])+'_beta_'+str(config["beta"]))
            U_inner_pred_np = U_inner_pred.cpu().detach().numpy().squeeze()
            ax = fig.add_subplot(3, 2, 1, projection='3d')
            ax.plot_trisurf(input_inner[:,0],input_inner[:,1], U_inner_pred_np,cmap='viridis', edgecolor='none');
            ax.set_title('U_inner_pred')

            loss_1_flat_np = loss_1_flat.cpu().detach().numpy().squeeze()
            ax = fig.add_subplot(3, 2, 2, projection='3d')
            ax.plot_trisurf(input_inner[:,0],input_inner[:,1], loss_1_flat_np,cmap='viridis', edgecolor='none');
            ax.set_title('loss_1')

            loss_2_flat_np = loss_2_flat.cpu().detach().numpy().squeeze()
            ax = fig.add_subplot(3, 2, 3, projection='3d')
            ax.plot_trisurf(input_inner[:,0],input_inner[:,1], loss_2_flat_np,cmap='viridis', edgecolor='none');
            ax.set_title('loss_2')

            loss_3_flat_np = loss_3_flat.cpu().detach().numpy().squeeze()
            ax = fig.add_subplot(3, 2, 4, projection='3d')
            ax.scatter(input_boundary[:,0],input_boundary[:,1], loss_3_flat_np);
            ax.set_title('loss_3')

            ax = fig.add_subplot(3, 2, 5, projection='3d')
            ax.plot_trisurf(input_inner[:,0],input_inner[:,1], np.abs(U_inner_pred_np-U_exact_inner.squeeze()),cmap='viridis', edgecolor='none');
            ax.set_title('U_pred_bias')

            U_boundary_pred_np= U_boundary_pred.cpu().detach().numpy().squeeze()
            ax = fig.add_subplot(3, 2, 6, projection='3d')
            ax.plot_trisurf(input_boundary[:,0],input_boundary[:,1],U_boundary_pred_np ,cmap='viridis', edgecolor='none');
            ax.set_title('U_boundary')
            plt.show()

    iter = iter+1
    Log['iter'].append(iter)
    Log['loss'].append(Loss.item())
    Log['loss_1'].append(loss_1.item())
    Log['loss_2'].append(loss_2.item()/alpha)
    Log['loss_3'].append(loss_3.item()/beta)
    Log['loss_exact'].append(loss_exact.item())




if not os.path.exists(config['path']):
    os.makedirs(config['path'])
NAME = os.path.join(config['path'],config['name'])
torch.save(model.state_dict(), NAME)
with open(os.path.join(config['path'],'config.json'),'w') as handle:
    json.dump(config,handle,indent=4, sort_keys=False)
with open(os.path.join(config['log'],'log.json'),'w') as handle:
    json.dump(log,handle,indent=4, sort_keys=False)

print(config)
