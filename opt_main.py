import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.colors as mplc
import cvxpy as cp

#from utils import set_random_seed, get_minibatches_idx
#from data import save_train_data, save_test_data, load_data_from_pickle

import numpy as np

import matplotlib.pyplot as plt
import os

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_tensor_type('torch.cuda.FloatTensor')

import torch.nn as nn
import torch.nn.functional as F

from utils import *
from train_utils import *
from ana_utils import *

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Neural Collapse test")
    parser.add_argument("--cluster_s", nargs='+',type=int, default=[3,3,4])
    parser.add_argument("--sample_s", nargs='+',type=int, default=[5000,4000,3000])
    parser.add_argument("--reg_z", type=float, default=0.005)
    parser.add_argument("--reg_b", type=float, default=0.01)
    parser.add_argument("--sd_b", type=str, default='ground_b')
    parser.add_argument("--sd_z", type=str, default='ground_z')
    
    argspar = parser.parse_args()
    k_s = argspar.cluster_s
    s_s = argspar.sample_s
    total_K = np.sum(k_s)
    
    if not os.path.exists('opt_results'):
        os.mkdir('opt_results')
    
    assert len(k_s) == len(s_s)
    n_list1 = []
    for i in range(len(k_s)):
        n_list1.extend([s_s[i] for j in range(k_s[i])])
        
    l = argspar.reg_z
    l_2 = argspar.reg_b
    
    
    ground_z,ground_b = solve_cvx(n_list1,l,l_2,total_K)
    print('ground truth z:')
    print(ground_z)
    print('ground truth b:')
    print(ground_b)
    print('saving results to opt_results folder')
    np.save('opt_results/'+argspar.sd_b+'.npy',ground_b)
    np.save('opt_results/'+argspar.sd_z+'.npy',ground_z)
    
    
    