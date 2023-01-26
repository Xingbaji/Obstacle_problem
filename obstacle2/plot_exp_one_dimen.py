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

device_num = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = device_num
device = torch.device('cuda:0')

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

# load_path = '/home/lehu.sx/PDE_experiments/results/Obstacle_one_dimen/1202_1148/'
load_path = '/home/lehu.sx/PDE_experiments/results/Obstacle_one_dimen/1202_1423/'

model_path = load_path + 'model.pth'
config_path = load_path + 'config.json'
with open(config_path, 'r') as handle:
    load_config = json.load(handle)
config = load_config
model = NormalMultiLayersModel(num_layers=config['depth'],hidden_size=config['width'],output_size = 1,input_size = 1)
model = model.to(device)
model.load_state_dict(torch.load(model_path))
print(model)

#np part
alpha = config["alpha"]
beta = config["beta"]
f = 0

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


input_inner,input_boundary,U_exact_inner,U_exact_boundary,g_inner = generate_uniform(config['num_linspace'])
input_inner_torch = torch.from_numpy(input_inner).to(device)
input_inner_torch.requires_grad = True
input_boundary_torch = torch.from_numpy(input_boundary).to(device)
U_exact_boundary_torch = torch.from_numpy(U_exact_boundary).to(device)
U_exact_inner_torch = torch.from_numpy(U_exact_inner).to(device)
g_torch = torch.from_numpy(g_inner).to(device)


diff = 1e-4
# input_inner_left_diff = input_inner -diff
input_inner_right_diff = input_inner + diff
# input_inner_left_diff = torch.from_numpy(input_inner_left_diff).to(device)
input_inner_right_diff = torch.from_numpy(input_inner_right_diff).to(device)
# x , w = leggauss_ab(input_inner.shape[0] -2 ,stepsize,1-stepsize)
iter = 0
min_loss_exact = 1000
min_loss_relative = 1000
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
loss_1 = torch.mean(loss_1_flat)
# Topk, index = torch.topk(torch.abs(loss_1_flat.squeeze()),10)
# loss_1_top = torch.mean(Topk)

loss_2_flat = F.relu(g_torch-U_inner_pred)**2 #contact part
loss_2 = alpha * torch.mean(loss_2_flat)
loss_3_flat = (U_boundary_pred-U_exact_boundary_torch)**2
loss_3 = beta * torch.mean(loss_3_flat)
Loss = loss_1 + loss_2 + loss_3
loss_exact = torch.mean(torch.abs(U_inner_pred - U_exact_inner_torch))
loss_relative = torch.mean(torch.abs((U_inner_pred[1:-1] - U_exact_inner_torch[1:-1])/U_exact_inner_torch[1:-1]))

print('Loss: {:.6f} Loss_1: {:.10f} Loss_2: {:.10f} Loss_3: {:.10f},Loss_exact: {:.6f},Loss_relative: {:.6f}'.format(
    Loss.item(),
    loss_1.item(),
    loss_2.item()/alpha,
    loss_3.item()/beta,
    loss_exact.item(),
    loss_relative.item(),
    ))

def plot_all(type=None):
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    fig.suptitle('iter'+str(iter)+'_'+config['loss_2_type']+'_alpha_'+str(config["alpha"])+'_beta_'+str(config["beta"]))
    U_inner_pred_np = U_inner_pred.cpu().detach().numpy().squeeze()
    ax = fig.add_subplot(2, 2, 1)
    plt.plot(input_inner, U_inner_pred_np);
    ax.set_title('U_inner_pred')

    loss_1_flat_np = loss_1_flat.cpu().detach().numpy().squeeze()
    ax = fig.add_subplot(2, 2, 2)
    plt.plot(input_inner, loss_1_flat_np);
    ax.set_title('loss_1')

    loss_2_flat_np = loss_2_flat.cpu().detach().numpy().squeeze()
    ax = fig.add_subplot(2, 2, 3)
    plt.plot(input_inner, loss_2_flat_np);
    ax.set_title('loss_2')

    ax = fig.add_subplot(2, 2, 4)
    plt.plot(input_inner, U_exact_inner.squeeze() - U_inner_pred_np);
    ax.set_title('U_pred_bias')

    plt.show()

U_inner_pred_np = U_inner_pred.cpu().detach().numpy().squeeze()
loss_1_flat_np = loss_1_flat.cpu().detach().numpy().squeeze()
loss_2_flat_np = loss_2_flat.cpu().detach().numpy().squeeze()
fig = plt.figure()
plt.plot(input_inner, U_inner_pred_np,'b',label='Estimated');
plt.plot(input_inner[::10], U_exact_inner[::10],'r--',label='True');
plt.plot(input_inner,g_inner,'g--',label='g')
plt.legend(loc='upper right')
plt.show()
print('')

plt.plot(input_inner, U_exact_inner.squeeze() - U_inner_pred_np,label=r'$u - u_{\phi}$');
plt.legend(loc='upper right')
plt.show()

log_path = load_path + 'log.json'
with open(log_path, 'r') as handle:
    Log = json.load(handle)
print('')
# plt.plot(Log['iter'],Log['loss']);plt.show()
log_loss = np.asarray(Log['loss'])
log_loss_1 = np.asarray(Log['loss_1'])
log_loss_2 = np.asarray(Log['loss_2'])
log_loss_3 = np.asarray(Log['loss_3'])
log_loss_exact = np.asarray(Log['loss_exact'])
# plt.plot(Log['iter'][:2000],np.log10(log_loss[:2000]));
# plt.plot(Log['iter'][:2000],np.log10(log_loss_1[:2000]));plt.show()
# plt.plot(Log['iter'][:2000],np.log10(log_loss_2[:2000]));plt.show()
# plt.plot(Log['iter'],np.log10(log_loss_exact));plt.show()

plt.plot(Log['iter'],np.log10(log_loss),label='Total loss');
plt.plot(Log['iter'],np.log10(log_loss_1),label=r'$loss_1$');
plt.legend(loc='upper right')
plt.show()
plt.plot(Log['iter'],np.log10(log_loss_2),label=r'$loss_2$');plt.legend(loc='upper right');plt.show()
plt.plot(Log['iter'],np.log10(log_loss_3),label=r'$loss_3$');plt.legend(loc='upper right');plt.show()
plt.plot(Log['iter'],np.log10(log_loss_exact),label=r'$\|u-u_{\phi}\|$');plt.legend(loc='upper right');plt.show()
print('')

