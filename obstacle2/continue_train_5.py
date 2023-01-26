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
    "path": '/home/sx/PDE_experiments/results/Obstacle_exp3/',
    "name": 'model.pth',
    "num_linspace": 201,
    "lr": 1e-4,
    "alpha": 5000, # project
    "beta": 5000, # boundary
    "num_iters": 5001,
    'visual': False,
    'depth': 10,
    'width':100,
    "loss_2_type": 'mean'
}
start_time = datetime.datetime.now().strftime('%m%d_%H%M')
config['path'] = os.path.join(config['path'],start_time)
if not os.path.exists(config['path']):
    os.makedirs(config['path'])

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
            x = torch.nn.functional.relu(x)**3
        x = self.fc[-1](x)
        return x

#Psi = np.minimum.reduce([X,Y,1-X,1-Y])


def g1_fun(input_xy):
    X = input_xy[:,0]
    Y = input_xy[:,1]
    G1 = (-1)*np.minimum.reduce([X,Y,1-X,1-Y])
    return G1


def k_fun(x):
    K = np.zeros_like(x)
    cond1 = (x>0) * (x<=1/6)
    cond2 = (x>1/6) * (x<=1/3)
    cond3 = (x>1/3) * (x<=1/2)
    cond4 = (x>1/2) * (x<=2/3)
    cond5 = (x>2/3) * (x<=5/6)
    cond6 = (x>5/6) * (x<=1)
    K[cond1] = 6*x[cond1]
    K[cond2] = 2*(1 - 3 * x[cond2])
    K[cond3] = 6*(x[cond3] - 1/3)
    K[cond4] = 2*(1-3*(x[cond4]-1/3))
    K[cond5] = 6*(x[cond5] - 2/3)
    K[cond6] = 2*(1-3*(x[cond6]-2/3))
    return K

def f_fun(input_xy):
    x = input_xy[:,0]
    y = input_xy[:,1]
    F = np.zeros_like(x)
    cond1 = (np.abs(x-y)<=0.1) * (x<=0.3)
    cond_tmp = np.invert(cond1)
    cond2 = (x<= (1-y)) * (cond_tmp)
    cond3 = (x> (1-y)) * (cond_tmp)
    F[cond1] = 300
    F[cond2] = -70*np.exp(y[cond2])*k_fun(x[cond2])
    F[cond3] = 15*np.exp(y[cond3])*k_fun(x[cond3])
    return F



#np part
alpha = config["alpha"]
beta = config["beta"]
f = 0
num_iters = config["num_iters"]

def generate_uniform(num_linsapce):
    X_axis = np.linspace(0,1,num_linsapce)
    Y_axis = np.linspace(0,1,num_linsapce)
    X,Y = np.meshgrid(X_axis,Y_axis)
    X_flat = np.reshape(X,[-1,1]).squeeze()
    Y_flat = np.reshape(Y,[-1,1]).squeeze()
    input_uniform = np.stack([X_flat,Y_flat],1).squeeze()
    input_boundary_x = np.concatenate([X[0,:],X[-1,:],X[:,0],X[:,-1]])
    input_boundary_y = np.concatenate([Y[0,:],Y[-1,:],Y[:,0],Y[:,-1]])
    input_boundary = np.stack([input_boundary_x,input_boundary_y],1)
    num_boundary = input_boundary.shape[0]
    # input_all = np.concatenate([input_boundary,input_uniform])

    K = k_fun(X_flat)
    F = f_fun(input_uniform)
    G1 = g1_fun(input_uniform)
    G2 = 0.2

    if config['visual']==True:
        fig = plt.figure()
        plt.plot(input_boundary[:,0],input_boundary[:,1],'x')
        plt.plot(input_uniform[:,0],input_uniform[:,1],'.')
        plt.title('sample points')
        fig.show()
        fig = plt.figure()
        plt.plot(input_uniform[:,0],K,'.')
        plt.title('K')
        fig.show()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title('f')
        ax.plot_trisurf(input_uniform[:,0],input_uniform[:,1], F,cmap='viridis', edgecolor='none');
        fig.show()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title('g')
        ax.plot_trisurf(input_uniform[:,0],input_uniform[:,1], G1,cmap='viridis', edgecolor='none');
        fig.show()

    # print('over')
    return input_uniform,input_boundary,F,G1,G2

input_inner,input_boundary,F,G1,G2 = generate_uniform(config['num_linspace'])
input_inner_torch = torch.from_numpy(input_inner).to(device)
input_inner_torch.requires_grad = True
input_boundary_torch = torch.from_numpy(input_boundary).to(device)
G1_torch = torch.from_numpy(G1).to(device)
F_torch = torch.from_numpy(F).to(device)
F_torch = F_torch.view([-1,1])
G1_torch = G1_torch.view([-1,1])
h = 0

load_path = '/home/sx/PDE_experiments/results/Obstacle_exp3/0126_0133/'
model_path = load_path + 'model.pth'
config_path = load_path + 'config.json'
with open(config_path, 'r') as handle:
    load_config = json.load(handle)

model = NormalMultiLayersModel(num_layers=load_config['depth'],hidden_size=load_config['width'],output_size = 1,input_size = 2)
model.to(device)
model.load_state_dict(torch.load(model_path))
print(model)

# model = NormalMultiLayersModel(num_layers=config['depth'],hidden_size=config['width'],output_size = 1,input_size = 2)
# model = model.to(device)
# print(model)
optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[5000,8000,10000], gamma=0.3)

Log = {'iter':[],'loss':[],'loss_1':[],'loss_2':[],'loss_3':[],'loss_4':[]}

iter = 0
while iter <= num_iters:
    optimizer.zero_grad()
    U_inner_pred = model(input_inner_torch) #[n.1]
    # U_boundary_pred = U_all[:num_boundary]
    U_boundary_pred  = model(input_boundary_torch)
    nabla_u = torch.autograd.grad(U_inner_pred, input_inner_torch, grad_outputs=torch.ones(U_inner_pred.size()).to(device), create_graph=True,retain_graph=True,only_inputs=True)[0] # [n,2]
    loss_1_flat = 1/2 * torch.sum(nabla_u**2,1,keepdim=True) - U_inner_pred*F_torch #all part
    loss_1 = torch.mean(loss_1_flat)

    loss_2_flat = torch.nn.functional.relu(G1_torch-U_inner_pred)**2 #contact part
    loss_2 = alpha * torch.mean(loss_2_flat)
    loss_3_flat = torch.nn.functional.relu(U_inner_pred-G2)**2 #contact part
    loss_3 = alpha * torch.mean(loss_3_flat)

    loss_4_flat = (U_boundary_pred-h)**2
    loss_4 = beta * torch.mean(loss_4_flat)
    Loss = loss_1 + loss_2 + loss_3 + loss_4
    Loss.backward()
    optimizer.step()
    scheduler.step()

    if iter % 10 == 0:
        print('Train iter: [{}/{}] Loss: {:.6f} Loss_1: {:.10f} Loss_2: {:.10f} Loss_3: {:.10f},Loss_4: {:.10f}'.format(
            iter,
            num_iters,
            Loss.item(),
            loss_1.item(),
            loss_2.item()/alpha,
            loss_3.item()/alpha,
            loss_4.item()/beta,
            ))

    # if iter == 5000:
    #     alpha = alpha * 100
    #     beta = beta * 100
    #     print(alpha,beta)



    if iter % 500 == 0  and iter>1:
        U_inner_pred_np = U_inner_pred.cpu().detach().numpy().squeeze()
        loss_1_flat_np = loss_1_flat.cpu().detach().numpy().squeeze()
        loss_2_flat_np = loss_2_flat.cpu().detach().numpy().squeeze()
        loss_3_flat_np = loss_3_flat.cpu().detach().numpy().squeeze()
        loss_4_flat_np = loss_4_flat.cpu().detach().numpy().squeeze()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(input_inner[:,0],input_inner[:,1], U_inner_pred_np,cmap='viridis', edgecolor='none');
        ax.set_title('U_pred_'+str(iter))
        plt.savefig(config['path'] + '/sol'+str(iter)+'.png')
        plt.close(fig)
        fig = plt.figure()
        fig.set_size_inches(10, 10)
        fig.suptitle('iter'+str(iter)+'_'+config['loss_2_type']+'_alpha_'+str(config["alpha"])+'_beta_'+str(config["beta"]))

        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.plot_trisurf(input_inner[:,0],input_inner[:,1], loss_1_flat_np,cmap='viridis', edgecolor='none');
        ax.set_title('loss_1')


        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.plot_trisurf(input_inner[:,0],input_inner[:,1], loss_2_flat_np,cmap='viridis', edgecolor='none');
        ax.set_title('loss_2')

        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.scatter(input_inner[:,0],input_inner[:,1], loss_3_flat_np);
        ax.set_title('loss_3')

        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.scatter(input_boundary[:,0],input_boundary[:,1], loss_4_flat_np);
        ax.set_title('loss_3')
        plt.savefig(config['path'] + '/loss'+str(iter)+'.png')
        plt.close(fig)
        # plt.show()
        fig = plt.figure()
        index_low = U_inner_pred_np < G1 + 1e-3
        coin_low = input_inner[index_low]
        index_high = G2  < U_inner_pred_np + 1e-3
        coin_high = input_inner[index_high]
        plt.plot(coin_low[:,0],coin_low[:,1],'r.')
        plt.plot(coin_high[:,0],coin_high[:,1],'bx')
        plt.savefig(config['path'] + '/coinset'+str(iter)+'.png')
        plt.close(fig)
    iter = iter+1
    Log['iter'].append(iter)
    Log['loss'].append(Loss.item())
    Log['loss_1'].append(loss_1.item())
    Log['loss_2'].append(loss_2.item()/alpha)
    Log['loss_3'].append(loss_3.item()/alpha)
    Log['loss_4'].append(loss_4.item()/beta)


if not os.path.exists(config['path']):
    os.makedirs(config['path'])
NAME = os.path.join(config['path'],config['name'])
torch.save(model.state_dict(), NAME)
with open(os.path.join(config['path'],'config.json'),'w') as handle:
    json.dump(config,handle,indent=4, sort_keys=False)
with open(os.path.join(config['path'],'log.json'),'w') as handle:
    json.dump(Log,handle,indent=4, sort_keys=False)

print(config)
log_loss = np.asarray(Log['loss'])
log_loss_1 = np.asarray(Log['loss_1'])
log_loss_2 = np.asarray(Log['loss_2'])
log_loss_3 = np.asarray(Log['loss_3'])
log_loss_4 = np.asarray(Log['loss_4'])

fig = plt.figure()
plt.plot(Log['iter'],np.log10(log_loss),label='Total loss');
plt.plot(Log['iter'],np.log10(log_loss_1),label=r'$loss_1$');
plt.legend(loc='upper right')
plt.savefig(config['path'] + '/Log_loss1'+str(iter)+'.png')
plt.close(fig)
# plt.show()
fig = plt.figure()
plt.plot(Log['iter'],np.log10(log_loss_2),label=r'$loss_2$');plt.legend(loc='upper right')
plt.savefig(config['path'] + '/Log_loss2'+str(iter)+'.png')
plt.close(fig)
fig = plt.figure()
plt.plot(Log['iter'],np.log10(log_loss_3),label=r'$loss_3$');plt.legend(loc='upper right');plt.show()
plt.savefig(config['path'] + '/Log_loss3'+str(iter)+'.png')
plt.close(fig)
fig = plt.figure()
plt.plot(Log['iter'],np.log10(log_loss_4),label=r'$loss4$');plt.legend(loc='upper right');plt.show()
plt.savefig(config['path'] + '/Log_loss4'+str(iter)+'.png')
plt.close(fig)
print('')
