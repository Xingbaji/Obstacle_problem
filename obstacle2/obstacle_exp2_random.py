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
    "path": '/home/lehu.sx/PDE_experiments/results/Obstacle_V3/',
    "name": 'model.pth',
    "num_linsapce": 401,
    "lr": 5e-5,
    "alpha": 5000, # project
    "beta": 5000, # boundary
    "num_iters": 10001,
    'visual': False,
    'depth': 10,
    'width':100,
    'num_inner': 40000,
    'num_boundary': 5000,
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


#np part
alpha = config["alpha"]
beta = config["beta"]
f = 0
num_iters = config["num_iters"]

def generate_sample(num_inner,num_boundary):
    X_inner = np.random.uniform(-2,2,num_inner)
    Y_inner = np.random.uniform(-2,2,num_inner)
    X_boundary_1 = np.random.uniform(-2,2,int(num_boundary/4))
    Y_boundary_1 = np.ones_like(X_boundary_1)*(-2)
    X_boundary_2 = np.random.uniform(-2,2,int(num_boundary/4))
    Y_boundary_2 = np.ones_like(X_boundary_2)*(2)
    Y_boundary_3 = np.random.uniform(-2,2,int(num_boundary/4))
    X_boundary_3 = np.ones_like(Y_boundary_3)*(-2)
    Y_boundary_4 = np.random.uniform(-2,2,int(num_boundary/4))
    X_boundary_4 = np.ones_like(Y_boundary_4)*(2)
    X_boundary = np.concatenate([X_boundary_1,X_boundary_2,X_boundary_3,X_boundary_4])
    Y_boundary = np.concatenate([Y_boundary_1,Y_boundary_2,Y_boundary_3,Y_boundary_4])
    input_boundary = np.stack([X_boundary,Y_boundary],1)
    input_inner = np.stack([X_inner,Y_inner],1)
    input_all = np.concatenate([input_boundary,input_inner])

    r = np.sqrt(input_all[:,0]**2+input_all[:,1]**2)
    r_tmp = np.sqrt(1-r**2)
    cond = r<=1
    g = np.ones_like(r)*(-1)
    g[cond] = r_tmp[cond]
    g = g[num_boundary:]

    r_star = 0.6979651482 #checked
    U_exact = np.zeros_like(r)
    cond1 = r<=r_star
    cond2 = r>r_star
    r_tmp2 = ((-1)*(r_star**2)*np.log(r/2))/np.sqrt(1-r_star**2)
    U_exact[cond1] = r_tmp[cond1]
    U_exact[cond2] = r_tmp2[cond2]

    U_exact_boundary = U_exact[:num_boundary]
    U_exact_inner = U_exact[num_boundary:]

    if np.isnan(np.max(U_exact)) == True or np.isnan(np.max(g)) == True: #避免nan
        raise NotImplementedError

    if config['visual']==True:
        fig = plt.figure()
        plt.plot(input_boundary[:,0],input_boundary[:,1],'x')
        plt.plot(input_inner[:,0],input_inner[:,1],'.')
        plt.title('sample points')
        fig.show()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title('g')
        ax.plot_trisurf(input_inner[:,0],input_inner[:,1], g,cmap='viridis', edgecolor='none');
        fig.show()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title('U')
        # points = ax.scatter(input_all[:,0],input_all[:,1], U_exact,c=U_exact, cmap='viridis', linewidth=0.5)
        ax.plot_trisurf(input_inner[:,0],input_inner[:,1], U_exact_inner,cmap='viridis', edgecolor='none');
        fig.show()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title('U')
        # points = ax.scatter(input_all[:,0],input_all[:,1], U_exact,c=U_exact, cmap='viridis', linewidth=0.5)
        ax.scatter(input_boundary[:,0],input_boundary[:,1], U_exact_boundary);
        fig.show()


    # print('over')
    U_exact_inner = np.reshape(U_exact_inner,[-1,1])
    U_exact_boundary = np.reshape(U_exact_boundary,[-1,1])
    g = np.reshape(g,[-1,1])
    return input_inner,input_boundary,U_exact_inner,U_exact_boundary,g

def generate_uniform(num_linsapce):
    X_axis = np.linspace(-2,2,num_linsapce)
    Y_axis = np.linspace(-2,2,num_linsapce)
    X,Y = np.meshgrid(X_axis,Y_axis)
    X_flat = np.reshape(X,[-1,1]).squeeze()
    Y_flat = np.reshape(Y,[-1,1]).squeeze()
    input_uniform = np.stack([X_flat,Y_flat],1).squeeze()

    # Input_boundary_x = np.concatenate([X[0,:],X[-1,:],X[:,0],X[:,-1]])
    # Input_boundary_y = np.concatenate([Y[0,:],Y[-1,:],Y[:,0],Y[:,-1]])
    # Input_boundary = np.stack([Input_boundary_x,Input_boundary_y],1)
    r = np.sqrt(X_flat**2+Y_flat**2)
    r_tmp = np.sqrt(1-r**2)

    r_star = 0.6979651482 #checked
    U_exact_uniform = np.zeros_like(X_flat)
    cond1 = r<=r_star
    cond2 = r>r_star
    r_tmp2 = ((-1)*(r_star**2)*np.log(r/2))/np.sqrt(1-r_star**2)
    U_exact_uniform[cond1] = r_tmp[cond1]
    U_exact_uniform[cond2] = r_tmp2[cond2]

    if np.isnan(np.max(U_exact_uniform)) == True: #避免nan
        raise NotImplementedError

    if config['visual']==True:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title('g')
        ax.plot_trisurf(X_flat,Y_flat, U_exact_uniform,cmap='viridis', edgecolor='none');
        fig.show()

    # print('over')
    return input_uniform,U_exact_uniform

input_uniform,U_exact_uniform = generate_uniform(201)
input_uniform_torch = torch.from_numpy(input_uniform).to(device)
U_uniform_pred_previous = np.zeros_like(U_exact_uniform)
num_inner = config['num_inner']
num_boundary  = config['num_boundary']

# model = NormalMultiLayersModel(num_layers=config['depth'],hidden_size=config['width'],output_size = 1,input_size = 2)
# model = model.to(device)
# print(model)
load_path = '/home/lehu.sx/PDE_experiments/results/Obstacle_V3/1127_1542/'
model_path = load_path + 'model.pth'
config_path = load_path + 'config.json'
with open(config_path, 'r') as handle:
    load_config = json.load(handle)

model = NormalMultiLayersModel(num_layers=load_config['depth'],hidden_size=load_config['width'],output_size = 1,input_size = 2)
model.to(device)
model.load_state_dict(torch.load(model_path))
print(model)

optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[4500,9000], gamma=0.1)
iter = 0
while iter <= num_iters:
    input_inner,input_boundary,U_exact_inner,U_exact_boundary,g_inner = generate_sample(num_inner,num_boundary)
    input_inner_torch = torch.from_numpy(input_inner).to(device)
    input_inner_torch.requires_grad = True
    input_boundary_torch = torch.from_numpy(input_boundary).to(device)
    U_exact_boundary_torch = torch.from_numpy(U_exact_boundary).to(device)
    U_exact_inner_torch = torch.from_numpy(U_exact_inner).to(device)
    g_torch = torch.from_numpy(g_inner).to(device)
    for i in range(100):
        optimizer.zero_grad()
        U_inner_pred = model(input_inner_torch) #[n.1]
        # U_boundary_pred = U_all[:num_boundary]
        U_boundary_pred  = model(input_boundary_torch)
        nabla_u = torch.autograd.grad(U_inner_pred, input_inner_torch, grad_outputs=torch.ones(U_inner_pred.size()).to(device), create_graph=True,retain_graph=True,only_inputs=True)[0] # [n,2]
        loss_1_flat = 1/2 * torch.sum(nabla_u**2,1,keepdim=True) - U_inner_pred*f #all part
        loss_1 = torch.mean(loss_1_flat)
        Topk, index = torch.topk(torch.abs(loss_1_flat.squeeze()),10)
        loss_1_top = torch.mean(Topk)

        loss_2_flat = F.relu(g_torch-U_inner_pred) #contact part
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
        # scheduler.step()
        loss_exact = torch.mean(torch.abs(U_inner_pred - U_exact_inner_torch))
        loss_exact_l2 = torch.mean(torch.abs(U_inner_pred - U_exact_inner_torch)**2)
        loss_exact_relative = torch.mean(torch.abs((U_inner_pred - U_exact_inner_torch)/U_exact_inner_torch))

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
        if iter % 100 == 0:
            with torch.no_grad():
                U_uniform_pred = model(input_uniform_torch)
                U_uniform_pred_np = U_uniform_pred.cpu().detach().numpy().squeeze()
                uniform_loss_l1 = np.mean(np.abs(U_uniform_pred_np - U_exact_uniform))
                uniform_loss_l2 = np.mean((U_uniform_pred_np - U_exact_uniform)**2)
                uniform_loss_relative = np.mean(np.abs((U_uniform_pred_np - U_exact_uniform)/U_exact_uniform))
                uniform_loss_change = np.mean(np.abs(U_uniform_pred_np - U_uniform_pred_previous))
                print('Train iter: [{}/{}] Uniform Lossl1: {:.6f} Loss_l2: {:.10f} Loss_relative: {:.10f}   Loss_change: {:.10f}'.format(
                    iter,
                    num_iters,
                    uniform_loss_l1,
                    uniform_loss_l2,
                    uniform_loss_relative,
                    uniform_loss_change,
                    ))
                U_uniform_pred_previous = U_uniform_pred_np

        # if iter % 1500 == 0 and iter >100:
        #     alpha = alpha * 10
        #     beta = beta * 10
        #     print('alpha: ',alpha,' beta: ',beta)
        #
        # if iter == 6000:
        #     # alpha = loss_1.item()/loss_2.item() * alpha
        #     # beta = loss_1.item()/loss_3.item() * alpha
        #     alpha = alpha * 10
        #     beta = beta * 10
        #     print('alpha: ',alpha,' beta: ',beta)
        #

        if iter % 5000 == 0 and iter>100:
            fig = plt.figure()
            fig.set_size_inches(10, 10)
            fig.suptitle('iter'+str(iter)+'_'+config['loss_2_type']+'_alpha_'+str(config["alpha"])+'_beta_'+str(config["beta"]))
            U_inner_pred_np = U_inner_pred.cpu().detach().numpy().squeeze()
            ax = fig.add_subplot(2, 2, 1, projection='3d')
            ax.plot_trisurf(input_inner[:,0],input_inner[:,1], U_inner_pred_np,cmap='viridis', edgecolor='none');
            ax.set_title('U_inner_pred')

            loss_1_flat_np = loss_1_flat.cpu().detach().numpy().squeeze()
            ax = fig.add_subplot(2, 2, 2, projection='3d')
            ax.plot_trisurf(input_inner[:,0],input_inner[:,1], loss_1_flat_np,cmap='viridis', edgecolor='none');
            ax.set_title('loss_1')

            loss_2_flat_np = loss_2_flat.cpu().detach().numpy().squeeze()
            ax = fig.add_subplot(2, 2, 3, projection='3d')
            ax.plot_trisurf(input_inner[:,0],input_inner[:,1], loss_2_flat_np,cmap='viridis', edgecolor='none');
            ax.set_title('loss_2')

            loss_3_flat_np = loss_3_flat.cpu().detach().numpy().squeeze()
            ax = fig.add_subplot(2, 2, 4, projection='3d')
            ax.scatter(input_boundary[:,0],input_boundary[:,1], loss_3_flat_np);
            ax.set_title('loss_3')
            plt.show()

            U_uniform_pred = model(input_uniform_torch)
            U_uniform_pred_np = U_uniform_pred.cpu().detach().numpy().squeeze()
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax.plot_trisurf(input_uniform[:,0],input_uniform[:,1], U_uniform_pred_np,cmap='viridis', edgecolor='none');
            ax.set_title('U_pred_uniform')
            ax = fig.add_subplot(1, 2, 2, projection='3d')
            ax.plot_trisurf(input_uniform[:,0],input_uniform[:,1], U_uniform_pred_np - U_exact_uniform,cmap='viridis', edgecolor='none');
            ax.set_title('bias_uniform')
            plt.show()

        iter = iter+1




if not os.path.exists(config['path']):
    os.makedirs(config['path'])
NAME = os.path.join(config['path'],config['name'])
torch.save(model.state_dict(), NAME)
with open(os.path.join(config['path'],'config.json'),'w') as handle:
    json.dump(config,handle,indent=4, sort_keys=False)

print(config)
