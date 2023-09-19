import torch
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import os

import torch
import torchvision
from torchvision import datasets, transforms
import pickle



def set_random_seed(seed):
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_minibatches_idx(n, minibatch_size, shuffle=True):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(math.ceil(n / minibatch_size)):
        minibatches.append(idx_list[minibatch_start: minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    return minibatches


def get_weighted_minibatches_idx(n, minibatch_size, weighted_idx, ratio, shuffle=True):
    idx_list = np.arange(n, dtype="int32")
    extra_idx = np.array(weighted_idx * (ratio - 1), dtype='int32')
    idx_list = np.concatenate((idx_list, extra_idx))
    if shuffle:
        np.random.shuffle(idx_list)
    n = len(idx_list)
    minibatches = []
    minibatch_start = 0
    for i in range(math.ceil(n / minibatch_size)):
        minibatches.append(idx_list[minibatch_start: minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    return minibatches

def save_train_data(data_t,name,cluster_info):
    if not os.path.exists('data'):
        os.mkdir('data')
    if data_t == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        trainset = datasets.CIFAR10(root='data/', train=True, download=True, transform=transform)
    elif data_t == 'fashion_mnist':
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(),
                                        transforms.Normalize((0.2860,), (0.3530,))])
        trainset = datasets.FashionMNIST('data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True,generator = torch.Generator(device='cuda'))
    train_data = []
    label_num = [0] * 10
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        label_idx = int(targets[0].item())
        if label_num[label_idx] >= cluster_info[label_idx]:
          continue
        train_data.append((inputs, targets))
        label_num[label_idx] += 1
    
    print('train label num', label_num)
    print('saving data to file data/'+name)
    with open('data/'+name+'.pickle', 'wb') as handle:
      pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
      handle.close()
    

def load_data_from_pickle(name):
    pickle_in_train = open("data/"+name,"rb")
    trainloader = pickle.load(pickle_in_train)
    pickle_in_train.close()
    return trainloader

def build_model(name,num_channel,lr,momentum,weight_decay):
    if name == 'ResNet18':
        model = ResNet18(color_channel=num_channel)
    elif name == 'VGG11':
        model = VGG('VGG11', color_channel=num_channel)
    elif name == 'VGG13':
        model = VGG('VGG13', color_channel=num_channel)
    else:
        print('wrong model option')
        model = None
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,  momentum=momentum,
                          weight_decay=weight_decay)

    return model, loss_function, optimizer