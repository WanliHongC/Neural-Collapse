import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.colors as mplc

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
    parser.add_argument("--dataset_t", type=str,\
                        default="cifar10", \
                        help='select dataset type cifar10 or fashion_mnist')
    parser.add_argument("--structure", type=int, default=1, \
                        help='1 or 2')
    parser.add_argument("--network", type=str, default="ResNet18", help='network to test with ResNet18, VGG11 or VGG13')
    parser.add_argument("--reg_b", type=float, default=0.01)
    parser.add_argument("--reg_h", type=float, default=1e-6)
    parser.add_argument("--reg_w", type=float, default=1e-3)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--ka", type=int, default=5)
    parser.add_argument("--nalist", type=int, default=5)
    
    
    
    argspar = parser.parse_args()
    
    cluster_info = {}

    # use CUDA?
    cluster_info = {}
    if argspar.structure == 1:
        for i in range(10):
            if i <= 3:
                cluster_info[i] = 5000
            elif i >= 7:
                cluster_info[i] = 3000
            else:
                cluster_info[i] = 4000
        
    else:
        for i in range(10):
            if i <= 3:
                cluster_info[i] = 5000
            elif i >= 6:
                cluster_info[i] = 1000
            else:
                cluster_info[i] = 3000
                
    data_name = '001_data_'+str(argspar.structure)+'_'+argspar.dataset_t
    if os.path.exists('data/'+data_name+'.pickle'):
        print('dataset already rendered')
    else:
        print('rendering dataset')
        save_train_data(argspar.dataset_t,data_name,cluster_info)
    lambda_s = [argspar.reg_h,argspar.reg_w,argspar.reg_b]
    trainloader = load_data_from_pickle(data_name+'.pickle')
    N = np.sum(list(cluster_info.values()))
    
    
    if argspar.dataset_t == 'cifar10':
        model, loss_function, optimizer = build_model(argspar.network,3,0.01,0.9,5e-4)
        save_dir = 'models/001_'+argspar.network+'_'+'train'+'_0'+str(argspar.structure)
    elif argspar.dataset_t == 'fashion_mnist':
        model, loss_function, optimizer = build_model(argspar.network,1,0.01,0.9,5e-4)
        save_dir = 'models/001_'+argspar.network+'_'+'f_train'+'_0'+str(argspar.structure)
    else:
        raise Exception("Invalid dataset choice, choose between either cifar10 or fashion_mnist")
        
    lr = 0.1
    batch_size = 128
    epochs = argspar.epoch
    if os.path.exists(save_dir+'/model_epoch='+str(epochs)):
        print('Pretrained model exists, skipping training phase')
    else:
        print('Training Start:')
        simple_train_batch(trainloader, model, loss_function, optimizer, epochs, lr, batch_size, lambda_s,save_dir,N)
        print('Training Complete!')
        print('------------------')
        print('Start computing NC_1 metric:')
    
    var_dict = {}
    
    if os.path.exists(save_dir+'/within_class_vari.pickle'):
        print('NC1 already computed, skipping computing phase')
        with open(save_dir+'/within_class_record.pickle', 'rb') as f: 
            var_dict = pickle.load(f)
    else:
        for name in range(1,epochs+1):
            path =save_dir+'/model_epoch='+str(name)
            if not os.path.exists(path):
                continue
            print('Computing for model at epoch '+str(name))
            model.load_state_dict(torch.load(path))
            model.eval()
            features,labels = get_features(trainloader,model)
            mean_features = {}
            centered_features = np.squeeze(features)
            global_mean = np.mean(features,axis=0)
            for i in range(10):
                mean_features[i] = np.mean(np.squeeze(features[np.where(labels==i),:]),axis=0)
                centered_features[np.where(labels==i)[0],:] -= mean_features[i]

            in_class_variance = 0
            #compute in-class variability
            for i in range(10):
                class_feature = centered_features[np.where(labels==i)[0],:]
                in_class_variance += ((class_feature).T@(class_feature))/N

            bet_class_variance = 0
            for i in range(10):
                diff = mean_features[i] - global_mean
                bet_class_variance += (diff.reshape(-1,1)@ diff.reshape(1,-1))*cluster_info[i]/N    
            
            
            
            variability = np.trace(in_class_variance@np.linalg.pinv(bet_class_variance,rcond=1e-10))/10    
            print('NC1 for model at epoch '+str(name)+': ',variability)
            var_dict[name] = variability
        

        with open(save_dir+'/within_class_vari.pickle', 'wb') as handle:
            pickle.dump(var_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()
            
    if not os.path.exists('plots'):
        os.mkdir('plots')
    plt.figure()
    plt.plot(list(var_dict.keys()),list(var_dict.values()))
    plt.xlabel('epoch number')
    plt.ylabel('$NC_1$')
    plt.yscale('log')
    plt.savefig('plots/001_'+argspar.network+'_'+argspar.dataset_t+'_'+str(argspar.structure)+'_NC1_plot.png')
    
    
    path =save_dir+'/model_epoch='+str(epochs)
    model.load_state_dict(torch.load(path))
    model.eval()
    weight = model.classifier.weight.cpu().detach().numpy()

    features,labels = get_features(trainloader,model)
    mean_features = {}
    centered_features = np.squeeze(features)
    global_mean = np.mean(features,axis=0)
    for i in range(10):
        mean_features[i] = np.mean(np.squeeze(features[np.where(labels==i),:]),axis=0)
    
    pre = np.array(list(mean_features.values())).dot(weight.T)
    plt.figure()
    plt.imshow((pre-np.min(pre)+0.01)/(np.max(pre)-np.min(pre)),norm=mplc.LogNorm(),cmap='gray')
    plt.savefig('plots/001_'+argspar.network+'_'+argspar.dataset_t+'_'+str(argspar.structure)+'_Z_plot.png')
    
    
    
    
    
    