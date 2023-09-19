import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

#from utils import set_random_seed, get_minibatches_idx
#from data import save_train_data, save_test_data, load_data_from_pickle

import matplotlib.pyplot as plt
import cvxpy as cp

import os

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_tensor_type('torch.cuda.FloatTensor')

import torch.nn as nn
import torch.nn.functional as F

from utils import set_random_seed, get_minibatches_idx


def get_features(trainloader, model):
    total_features = []
    total_labels = []
    minibatches_idx = get_minibatches_idx(len(trainloader), minibatch_size=128,
                                        shuffle=False)
    for minibatch in minibatches_idx:
        inputs = torch.Tensor(np.array([list(trainloader[x][0].cpu().numpy()) for x in minibatch]))
        targets = torch.Tensor(np.array([list(trainloader[x][1].cpu().numpy()) for x in minibatch]))
        inputs, targets = Variable(inputs.cuda()).squeeze(1), Variable(targets.cuda()).squeeze()
        _,features = model.forward(inputs)
        total_features.extend(features.cpu().data.numpy().tolist())
        total_labels.extend(targets.cpu().data.numpy().tolist())
    total_features = np.array(total_features)
    total_labels = np.array(total_labels)
    return total_features,total_labels
    
   
def compute_pairwise_angle(F):
    inner = F.dot(F.T)
    norm = np.linalg.norm(F,axis=1)
    cos = inner / norm.reshape(-1,1)/norm.reshape(1,-1)
    return cos

def solve_cvx(n_list,l,l_2,K):
    K = len(n_list)
    Y = np.eye(K)

    scaling = np.sqrt(np.diag(n_list))
    W = cp.Variable(shape=(K,K))
    b = cp.Variable(shape=(K,1))
    ones = np.ones((1,K))
    N = np.sum(n_list)

    objective = cp.Minimize(cp.sum(cp.multiply(cp.log_sum_exp(W+(b@ones),axis=0)-cp.sum(cp.multiply(Y,W+(b@ones)),axis=0),n_list))/N+l*cp.norm(W@scaling,'nuc')+l_2*cp.square(cp.norm(b,2)))
    prob = cp.Problem(objective)
    result = prob.solve()
    z = W.value
    b = (b.value).reshape(-1)
    return z,b