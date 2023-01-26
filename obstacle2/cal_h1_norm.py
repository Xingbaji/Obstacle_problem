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

def u(x):
    if (x>=0) and (x< 1/(2*math.sqrt(2))):
        return ((100 - 50 * math.sqrt(2)) * x)**2
    if (x>= 1/(2*math.sqrt(2))) and (x< 0.5):
        return (100 * x * (1 - x) - 12.5)**2
    if (x>=0.5) and (x<1- 1/(2*math.sqrt(2))):
        return (100 * x * (1 - x) - 12.5)**2
    if (x>= ( 1 - 1/(2*math.sqrt(2)))) and (x<=1):
        return ((100 - 50 * math.sqrt(2)) * (1 - x))**2
def grad_u(x):
    if (x>=0) and (x< 1/(2*math.sqrt(2))):
        return ((100 - 50 * math.sqrt(2)))**2
    if (x>= 1/(2*math.sqrt(2))) and (x< 0.5):
        return (100 * (1 - 2*x) - 12.5)**2
    if (x>=0.5) and (x<1- 1/(2*math.sqrt(2))):
        return (100 * (1 - 2*x) - 12.5)**2
    if (x>= ( 1 - 1/(2*math.sqrt(2)))) and (x<=1):
        return ((100 - 50 * math.sqrt(2)) * (-1))**2

from scipy.integrate import quad

res1, err1 = quad(u, 0, 1)
res2, err2 = quad(grad_u, 0, 1)
print(res1,res2)
# |u|_{H1} =  28.323046032801834

H = np.array([[1,1/2],[1/2,1/3]])
print(np.linalg.inv(H))




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

