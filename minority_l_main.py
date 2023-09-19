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
    parser.add_argument("--na",type=int, default=500)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--ka", type=int, default=5)
    parser.add_argument("--network", type=str, default="ResNet18", help='network to test with ResNet18, VGG11 or VGG13')
    parser.add_argument("--reg_b", type=float, default=0.01)
    parser.add_argument("--reg_z_l", nargs='+',type=float, default = [0.0027,0.0030,0.0033,0.0039,0.0045,0.0051,0.0057,0.0063,0.0069,0.0075])
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--no_gpu", action='store_true', help="run model on CPU")
    
    argspar = parser.parse_args()
    na = argspar.na
    ka = argspar.ka
    reg_z_l = argspar.reg_z_l
    print('testing with l:',reg_z_l)
    cluster_info = {}
    epochs = argspar.epoch
    # use CUDA?
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()
    z_list = []
    for l in reg_z_l:
        cluster_info = {}
        for i in range(10):
            if i <= argspar.ka-1:
                cluster_info[i] = na
            else:
                cluster_info[i] = argspar.nb
        
        data_name = '234_data_ka={}_na={}_nb={}'.format(argspar.ka,na,argspar.nb)
        if os.path.exists('data/'+data_name+'.pickle'):
            print('dataset already rendered for na={}'.format(na))
        else:
            print('rendering dataset for na={}'.format(na))
            save_train_data(argspar.dataset_t,data_name,cluster_info)
        lambda_s = [l/20,l*5,argspar.reg_b]
        trainloader = load_data_from_pickle(data_name+'.pickle')
        N = np.sum(list(cluster_info.values()))
    
    
        if argspar.dataset_t == 'cifar10':
            model, loss_function, optimizer = build_model(argspar.network,3,0.01,0.9,0)
        elif argspar.dataset_t == 'fashion_mnist':
            model, loss_function, optimizer = build_model(argspar.network,1,0.01,0.9,0)
        else:
            raise Exception("Invalid dataset choice, choose between either cifar10 or fashion_mnist")

        lr = 0.1
        batch_size = 128
        save_dir = 'models/003_{}_ka={}_na={}_nb={}_l={}'.format(argspar.network,argspar.ka,na,argspar.nb,l)
        if os.path.exists(save_dir+'/network_weight'):
            print('Pretrained model exists, skipping training phase')
        else:
            print('Training Start:')
            simple_train_batch_warm(trainloader, model, loss_function, optimizer, epochs, lr, batch_size, lambda_s,save_dir,N)
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
    r=2
    c=5
    fig,axs = plt.subplots(r,c,sharex=True,sharey=True)
    fig.set_figheight(4)
    fig.set_figwidth(4/r*c+1)
    
    vmin,vmax = np.min(data_array),np.max(data_array)
    

    for k in len(reg_z_l):
        l = sorted(reg_z_l)[k]
        real_k = na_list.index(l)
        data = data_array[real_k,:,:]
       # pos = axs[k//c,k%c].imshow((data.T-vmin)/(vmax-vmin),vmin=0,vmax=1,cmap='gray')
        pos = axs[k//c,k%c].imshow(data.T,cmap='gray',norm = colors.SymLogNorm(linthresh=0.03,vmin=vmin,vmax=vmax))
        axs[k//c,k%c].set_yticks([0,ka])
        axs[k//c,k%c].set_xticks([0,ka])
        axs[k//c,k%c].set_title('$l={}$'.format(l))

        axs[k//c,k%c].plot([ka-0.5,ka-0.5],[-0.5,9.5],c='w',linestyle='--')
        axs[k//c,k%c].plot([-0.5,9.5],[ka-0.5,ka-0.5],c='w',linestyle='--')

    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(pos, cax=cbar_ax)
    print('saving plot to plots/003_{}_ka={}_nb={}_na={}_z_comparison_plot'.format(argspar.network,argspar.ka,argspar.nb,na))
    plt.savefig('plots/003_{}_ka={}_nb={}_na={}_z_comparison_plot.png'.format(argspar.network,argspar.ka,argspar.nb,na))
    
    
    
    
    
    
    
    
    
    
    
    