import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.colors as mplc
from matplotlib import colors 

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
    set_random_seed(666)
    parser = argparse.ArgumentParser(description="Neural Collapse test")
    parser.add_argument("--dataset_t", type=str,\
                        default="cifar10", \
                        help='select dataset type cifar10 or fashion_mnist')
    parser.add_argument("--network", type=str, default="ResNet18", help='network to test with ResNet18, VGG11 or VGG13')
    parser.add_argument("--reg_b", type=float, default=0.01)
    parser.add_argument("--ln_pro", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--ka", type=int, default=5)
    parser.add_argument("--nalist", nargs='+', type=int, default=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000])
    parser.add_argument("--ratio", type=float,default=2)
    
    argspar = parser.parse_args()
    na_list = argspar.nalist
    ka = argspar.ka
    print('testing with :',na_list)
    cluster_info = {}
    epochs = argspar.epoch
    z_list = []
    for na in na_list:
        cluster_info = {}
        nb = na//argspar.ratio
        for i in range(10):
            if i <= argspar.ka-1:
                cluster_info[i] = na
            else:
                cluster_info[i] = nb
        N = np.sum(list(cluster_info.values()))
        data_name = '234_data_ka={}_na={}_nb={}'.format(argspar.ka,na,nb)
        if os.path.exists('data/'+data_name+'.pickle'):
            print('dataset already rendered for na={}'.format(na))
        else:
            print('rendering dataset for na={}'.format(na))
            save_train_data(argspar.dataset_t,data_name,cluster_info)
        l = argspar.ln_pro/N
        lambda_s = [l/20,l*5,argspar.reg_b]
        trainloader = load_data_from_pickle(data_name+'.pickle')
        N = np.sum(list(cluster_info.values()))
    
    
        if argspar.dataset_t == 'cifar10':
            model, loss_function, optimizer = build_model(argspar.network,3,0.01,0.9,5e-4)
        elif argspar.dataset_t == 'fashion_mnist':
            model, loss_function, optimizer = build_model(argspar.network,1,0.01,0.9,5e-4)
        else:
            raise Exception("Invalid dataset choice, choose between either cifar10 or fashion_mnist")

        lr = 0.1
        batch_size = 128
        save_dir = 'models/004_{}_ka={}_na={}_r={}'.format(argspar.network,argspar.ka,na,argspar.ratio)
        if os.path.exists(save_dir+'/network_weight'):
            print('Pretrained model exists, skipping training phase')
        else:
            print('Training Start:')
            simple_train_batch_2(trainloader, model, loss_function, optimizer, epochs, lr, batch_size, lambda_s,save_dir,N)
            print('Training Complete for na={}'.format(na))
            print('------------------')
        
        
        model.load_state_dict(torch.load(save_dir+'/network_weight'))
        model.eval()
        features,labels = get_features(trainloader,model)
        mean_features = {}
        for i in range(10):
            mean_features[i] = np.mean(np.squeeze(features[np.where(labels==i),:]),axis=0)
        weight = model.classifier.weight.cpu().detach().numpy()
        bias = model.classifier.bias.cpu().detach().numpy()
        pre = np.array(list(mean_features.values())).dot(weight.T)
        np.save(save_dir+'/z_data.npy',pre)
        np.save(save_dir+'/b_data.npy',bias)
        print('bias:',bias)
        print('pre:',pre)
        z_list.append(pre)
        
    data_array = np.array(z_list)        
    print('Generating plots:')
    fig = plt.figure()
    fig.set_figheight(3)
    fig.set_figwidth(5) 

    angles = np.zeros((3,data_array.shape[0]))
    count = 0
    for i in range(data_array.shape[0]):
        z = data_array[i,:,:]
        angle = compute_pairwise_angle(z.T)
        angles[0,count] = np.mean(angle[0:ka,0:ka][np.triu_indices(ka,k=1)])
        angles[1,count] = np.mean(angle[ka:10,ka:10][np.triu_indices(ka,k=1)])
        angles[2,count] = np.mean(angle[0:ka,ka:10])
        count += 1


    linestyle = ['-.',':','--']
    c = ['r','b','k']

    for i in range(3):
        plt.plot(na_list,angles.T[:,i],c=c[i],linestyle=linestyle[i],linewidth=2.5)

    plt.xlabel('$n_A$',size=12)
    plt.ylabel('cosine angle',size=12)


    plt.plot([np.min(na_list),np.max(na_list)],[-1/9,-1/9],c='y')
    plt.xlim([np.min(na_list),np.max(na_list)])
    plt.legend(['A','B','AB','ETF angle'])
    print('saving results to plots/limit.png')
    plt.savefig('plots/limit.png',bbox_inches='tight')
    plt.show()

    
    
    
    
    
    
    
    
    
    
    