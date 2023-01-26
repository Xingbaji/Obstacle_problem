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
import math
torch.set_default_tensor_type('torch.DoubleTensor')

device_num = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = device_num
device = torch.device('cuda:0')
config = {
    "path": '/home/lehu.sx/PDE_experiments/results/Obstacle_one_dimen/',
    "name": 'model.pth',
    "num_linspace": 5001,
    "lr": 5e-4,
    "alpha": 5000, # project
    "beta": 5000, # boundary
    "num_iters": 3001,
    'visual': False,
    'depth': 8,
    'width': 80,
    "loss_2_type": 'mean'
}
start_time = datetime.datetime.now().strftime('%m%d_%H%M')
config['path'] = os.path.join(config['path'],start_time)

def leggauss_ab(n=96, a=-1.0, b=1.0):
    assert(n>0)
    x,w = np.polynomial.legendre.leggauss(n)
    x = (b-a) * 0.5 * x+(b+a) * 0.5
    w = w * (b-a) * 0.5
    return x,w


class NormalMultiLayersModel(nn.Module):
    def __init__(self,num_layers = 5, hidden_size = 50,output_size = 1,input_size = 1):
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
            x = F.relu(x)
            x = layernorm(x)
            x = F.relu(x)**3
        x = self.fc[-1](x)
        return x


#np part
alpha = config["alpha"]
beta = config["beta"]
f = 0
num_iters = config["num_iters"]

def cal_u_exact(x):
    # x = np.linspace(0,1,100)
    u = np.zeros_like(x)
    cond1 = (x>=0) * (x< 1/(2*math.sqrt(2)))
    cond2 = (x>= 1/(2*math.sqrt(2))) * (x< 0.5)
    cond3 = (x>=0.5) * (x<1- 1/(2*math.sqrt(2)))
    cond4 = (x>= ( 1 - 1/(2*math.sqrt(2)))) * (x<=1)
    x1 = x[cond1]
    x2 = x[cond2]
    x3 = x[cond3]
    x4 = x[cond4]
    u[cond1] = (100 - 50 * math.sqrt(2)) *x1
    u[cond2] = 100 * x2 * (1 - x2) - 12.5
    u[cond3] = 100 * x3 * (1 - x3) - 12.5
    u[cond4] = (100 - 50 * math.sqrt(2)) * (1 - x4)
    # if config['visual'] == True:
    #     plt.plot(x,u);plt.show()
    return u

def cal_g(x):
    # x = np.linspace(0,1,100)
    g = np.zeros_like(x)
    cond1 = (x>=0) * (x<0.25)
    cond2 = (x>= 0.25) * (x< 0.5)
    cond3 = (x>=0.5) * (x < 0.75)
    cond4 = (x>=0.75) * (x <= 1.0)
    x1 = x[cond1]
    x2 = x[cond2]
    x3 = x[cond3]
    x4 = x[cond4]
    g[cond1] = 100 * x1**2
    g[cond2] = 100 * x2 * (1 - x2) - 12.5
    g[cond3] = 100 * x3 * (1 - x3) - 12.5
    g[cond4] = 100 * (1-x4)**2
    # if config['visual'] == True:
    #     plt.plot(x,g);plt.show()
    return g

def generate_uniform(num_linsapce=100):
    input_uniform = np.linspace(0,1,num_linsapce)[1:-1]
    input_uniform = np.reshape(input_uniform,[-1,1])
    U_exact_uniform = cal_u_exact(input_uniform)

    input_boundary = np.asarray([[0],[1]],dtype='float64')
    U_exact_boundary = np.zeros_like(input_boundary)

    g = cal_g(input_uniform)

    # U_exact_uniform = np.reshape(U_exact_uniform,[-1,1])
    # U_exact_boundary = np.reshape(U_exact_boundary,[-1,1])
    # g = np.reshape(g,[-1,1])

    return input_uniform,input_boundary,U_exact_uniform,U_exact_boundary,g

def generate_gauss(number = 100):
    X_gauss,W = leggauss_ab(number,0,1)
    X_gauss = np.reshape(X_gauss,[-1,1])
    W = np.reshape(W,[-1,1])
    U_exact = cal_u_exact(X_gauss)

    input_boundary = np.asarray([[0],[1]],dtype='float64')
    U_exact_boundary = np.zeros_like(input_boundary)

    g = cal_g(X_gauss)

    # U_exact_uniform = np.reshape(U_exact_uniform,[-1,1])
    # U_exact_boundary = np.reshape(U_exact_boundary,[-1,1])
    # g = np.reshape(g,[-1,1])

    return X_gauss,input_boundary,U_exact,U_exact_boundary,g,W

input_inner,input_boundary,U_exact_inner,U_exact_boundary,g_inner,W = generate_gauss(config['num_linspace'])
input_inner_torch = torch.from_numpy(input_inner).to(device)
input_inner_torch.requires_grad = True
input_boundary_torch = torch.from_numpy(input_boundary).to(device)
U_exact_boundary_torch = torch.from_numpy(U_exact_boundary).to(device)
U_exact_inner_torch = torch.from_numpy(U_exact_inner).to(device)
g_torch = torch.from_numpy(g_inner).to(device)
W_torch = torch.from_numpy(W).to(device)

input_uniform,_,U_exact_uniform,_,_ = generate_uniform(config['num_linspace'])
input_uniform_torch = torch.from_numpy(input_uniform).to(device)
U_exact_uniform_torch = torch.from_numpy(U_exact_uniform).to(device)

model = NormalMultiLayersModel(num_layers=config['depth'],hidden_size=config['width'],output_size = 1,input_size = 1)
model = model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[1000], gamma=0.3)

diff = 1e-4
# input_inner_left_diff = input_inner -diff
input_inner_right_diff = input_inner + diff
# input_inner_left_diff = torch.from_numpy(input_inner_left_diff).to(device)
input_inner_right_diff = torch.from_numpy(input_inner_right_diff).to(device)
# x , w = leggauss_ab(input_inner.shape[0] -2 ,stepsize,1-stepsize)
iter = 0
min_loss_exact = 1000
min_loss_relative = 1000
Log = {'iter':[],'loss':[],'loss_1':[],'loss_2':[],'loss_3':[],'loss_exact':[]}

while iter <= num_iters:
    optimizer.zero_grad()
    U_inner_pred = model(input_inner_torch) #[n.1]
    # U_inner_pred_left = model(input_inner_left_diff)
    U_inner_pred_right = model(input_inner_right_diff)
    U_boundary_pred  = model(input_boundary_torch)
    # diff_u = (U_inner_pred[1:] - U_inner_pred[:-1])/(stepsize) # 中心差分
    diff_u = (U_inner_pred_right - U_inner_pred)/diff # 中心差分
    # nabla_u = torch.autograd.grad(U_inner_pred, input_inner_torch, grad_outputs=torch.ones(U_inner_pred.size()).to(device), create_graph=True,retain_graph=True,only_inputs=True)[0] # [n,2]
    # Delta_u = torch.autograd.grad(nabla_u, input_inner_torch, grad_outputs=torch.ones(U_inner_pred.size()).to(device), create_graph=True,retain_graph=True,only_inputs=True)[0] # [n,2]
    loss_1_flat = 1/2 * diff_u**2
    # loss_1_flat = 1/2 * nabla_u**2
    loss_1 = torch.sum(W_torch *loss_1_flat)
    # Topk, index = torch.topk(torch.abs(loss_1_flat.squeeze()),10)
    # loss_1_top = torch.mean(Topk)

    loss_2_flat = F.relu(g_torch-U_inner_pred)**2 #contact part
    loss_2 = alpha * torch.mean(loss_2_flat)
    loss_3_flat = (U_boundary_pred-U_exact_boundary_torch)**2
    loss_3 = beta * torch.mean(loss_3_flat)
    Loss = loss_1 + loss_2 + loss_3
    loss_exact = torch.mean(torch.abs(U_inner_pred - U_exact_inner_torch))
    loss_relative = torch.mean(torch.abs((U_inner_pred[1:-1] - U_exact_inner_torch[1:-1])/U_exact_inner_torch[1:-1]))

    if iter % 10 == 0:
        print('Train iter: [{}/{}] Loss: {:.6f} Loss_1: {:.10f} Loss_2: {:.10f} Loss_3: {:.10f},Loss_exact: {:.6f},Loss_relative: {:.6f}'.format(
            iter,
            num_iters,
            Loss.item(),
            loss_1.item(),
            loss_2.item()/alpha,
            loss_3.item()/beta,
            loss_exact.item(),
            loss_relative.item(),
            ))

    if iter % 500 == 0:
        with torch.no_grad():
            U_uniform_pred = model(input_uniform_torch)
            loss_exact_uniform = torch.mean(torch.abs(U_uniform_pred - U_exact_uniform_torch))
            loss_relative_uniform = torch.mean(torch.abs((U_uniform_pred[1:-1] - U_exact_uniform_torch[1:-1])/U_exact_uniform_torch[1:-1]))
            print('test on uniform: ',loss_exact_uniform.item(),loss_relative_uniform.item())

    if loss_relative.item() < min_loss_relative:
        min_loss_exact = loss_exact.item()
        min_loss_relative = loss_relative.item()
        # if loss_exact < 0.01:
        #     if not os.path.exists(config['path']):
        #         os.makedirs(config['path'])
        #     NAME = os.path.join(config['path'],config['name'])
        #     torch.save(model.state_dict(), NAME)
        #     with open(os.path.join(config['path'],'config.json'),'w') as handle:
        #         json.dump(config,handle,indent=4, sort_keys=False)
        # print(min_loss_exact,min_loss_relative)

        # fig = plt.figure()
        # fig.set_size_inches(10, 10)
        # fig.suptitle('iter'+str(iter)+'_'+config['loss_2_type']+'_alpha_'+str(config["alpha"])+'_beta_'+str(config["beta"]))
        # U_inner_pred_np = U_inner_pred.cpu().detach().numpy().squeeze()
        # ax = fig.add_subplot(2, 2, 1)
        # plt.plot(input_inner, U_inner_pred_np);
        # ax.set_title('U_inner_pred')
        #
        # loss_1_flat_np = loss_1_flat.cpu().detach().numpy().squeeze()
        # ax = fig.add_subplot(2, 2, 2)
        # plt.plot(input_inner, loss_1_flat_np);
        # ax.set_title('loss_1')
        #
        # loss_2_flat_np = loss_2_flat.cpu().detach().numpy().squeeze()
        # ax = fig.add_subplot(2, 2, 3)
        # plt.plot(input_inner, loss_2_flat_np);
        # ax.set_title('loss_2')
        #
        # ax = fig.add_subplot(2, 2, 4)
        # plt.plot(input_inner, U_exact_inner.squeeze() - U_inner_pred_np);
        # ax.set_title('U_pred_bias')
        #
        # plt.show()
        # print('')

        # if iter == 2000:
        #     print('')
        #     ## test nabla
        #     diff_u = (U_inner_pred[2:] - U_inner_pred[:-2])/(2*stepsize)
        #     diff_u_test = diff_u[:24]
        #     x_test = input_inner_torch[1:25]
        #     nabla_u_true =(100 - 50*math.sqrt(2))*x_test
        #     nabla_u_torch =  torch.autograd.grad(U_inner_pred, input_inner_torch, grad_outputs=torch.ones(U_inner_pred.size()).to(device), create_graph=True)[0]
        #     nabla_u_torch_test = nabla_u_torch[1:25]

    Loss.backward()
    optimizer.step()
    scheduler.step()
    iter = iter+1
    Log['iter'].append(iter)
    Log['loss'].append(Loss.item())
    Log['loss_1'].append(loss_1.item())
    Log['loss_2'].append(loss_2.item()/alpha)
    Log['loss_3'].append(loss_3.item()/beta)
    Log['loss_exact'].append(loss_exact.item())



print(min_loss_exact,min_loss_relative)
print(config)

# if not os.path.exists(config['path']):
#     os.makedirs(config['path'])
# # NAME = os.path.join(config['path'],config['name'])
# # torch.save(model.state_dict(), NAME)

# with open(os.path.join(config['path'],'log.json'),'w') as handle:
#     json.dump(Log,handle,indent=4, sort_keys=False)

# log_loss = np.asarray(Log['loss'])
# log_loss_1 = np.asarray(Log['loss_1'])
# log_loss_2 = np.asarray(Log['loss_2'])
# log_loss_3 = np.asarray(Log['loss_3'])
# log_loss_exact = np.asarray(Log['loss_exact'])
# plt.plot(Log['iter'],np.log10(log_loss));
# plt.plot(Log['iter'],np.log10(log_loss_1));plt.show()
# plt.plot(Log['iter'],np.log10(log_loss_2));plt.show()
# plt.plot(Log['iter'],np.log10(log_loss_3));plt.show()
# plt.plot(Log['iter'],np.log10(log_loss_exact));plt.show()
#
