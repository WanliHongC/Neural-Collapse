import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import os

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_tensor_type('torch.cuda.FloatTensor')

import torch.nn as nn
import torch.nn.functional as F

from utils import set_random_seed, get_minibatches_idx


# reference: https://github.com/kuangliu/pytorch-cifar
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, color_channel=3, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(color_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out2 = out.view(out.size(0), -1)
        out3 = self.classifier(out2)
        return out3,out2

    def get_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

class ResNet_bf(nn.Module):
    def __init__(self, block, num_blocks, color_channel=3, num_classes=10):
        super(ResNet_bf, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(color_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512*block.expansion, num_classes, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out2 = out.view(out.size(0), -1)
        out3 = self.classifier(out2)
        return out3,out2

    def get_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def ResNet18(color_channel=3, num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], color_channel=color_channel, num_classes=num_classes)

def ResNet18_bf(color_channel=3, num_classes=10):
    return ResNet_bf(BasicBlock, [2, 2, 2, 2], color_channel=color_channel, num_classes=num_classes)

def ResNet34(color_channel=3, num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], color_channel=color_channel, num_classes=num_classes)


def ResNet50(color_channel=3, num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], color_channel=color_channel, num_classes=num_classes)


def ResNet101(color_channel=3, num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], color_channel=color_channel, num_classes=num_classes)


def ResNet152(color_channel=3, num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], color_channel=color_channel, num_classes=num_classes)


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, color_channel=3, num_classes=10):
        super(VGG, self).__init__()
        self.color_channel = color_channel
        self.features = self._make_layers(cfg[vgg_name])
        self.fc1 = nn.Linear(512, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.classifier = nn.Linear(4096, num_classes)

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.color_channel
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.features(x)
        out2 = out1.view(out1.size(0), -1)
        out3 = F.relu(self.bn1(self.fc1(out2)))
        out4 = F.relu(self.fc2(out3))
        out5 = self.classifier(out4)
        return out5,out4

    def get_features(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.bn1(self.fc1(out)))
        out = F.relu(self.fc2(out))
        return out


import torch
import torchvision
from torchvision import datasets, transforms
import pickle


    
   
def build_model(name,num_channel,lr,momentum,weight_decay):
    if name == 'ResNet18':
        model = ResNet18(color_channel=num_channel)
    elif name == 'VGG11':
        model = VGG('VGG11', color_channel=num_channel)
    elif name == 'VGG13':
        model = VGG('VGG13', color_channel=num_channel)
    elif name == 'ResNet18_bf':
        model = ResNet18_bf(color_channel=num_channel)
    else:
        print('wrong model option')
        model = None
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,  momentum=momentum,
                          weight_decay=weight_decay)

    return model, loss_function, optimizer



def simple_train_batch(trainloader, model, loss_function, optimizer, epochs, lr, batch_size, lambda_s,save_dir,N):
    if os.path.exists(save_dir):
        pass
    else:
        os.mkdir(save_dir)
    model.train()
    for epoch in range(epochs):
        if epoch == int(epochs / 6):
            for g in optimizer.param_groups:
                g['lr'] = lr / 10
            print('divide current learning rate by 10')
        elif epoch == int(epochs /3):
            for g in optimizer.param_groups:
                g['lr'] = lr / 100
            print('divide current learning rate by 10')
        total_loss = 0
        feature_loss = 0
        minibatches_idx = get_minibatches_idx(len(trainloader), minibatch_size=batch_size,
                                              shuffle=True)
        
        for minibatch in minibatches_idx:
            inputs = torch.Tensor(np.array([list(trainloader[x][0].cpu().numpy()) for x in minibatch]))
            targets = torch.Tensor(np.array([list(trainloader[x][1].cpu().numpy()) for x in minibatch]))
            inputs, targets = Variable(inputs.cuda()).squeeze(1), Variable(targets.long().cuda()).squeeze()
            optimizer.zero_grad()
            outputs, features_f_re = model(inputs)
            outputs = outputs.squeeze()
            weight_loss = lambda_s[1]*torch.sum(model.classifier.weight**2)
            feature_loss =  lambda_s[0]*torch.sum(features_f_re**2)/len(targets)*N
            bias_loss = lambda_s[2]*torch.sum(model.classifier.bias**2)
            prediction_loss = loss_function(outputs, targets)
            loss = prediction_loss + weight_loss + bias_loss + feature_loss
            total_loss += loss
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0 or (epoch+1) <= 5:
            torch.save(model.state_dict(), save_dir+'/model_epoch='+str(epoch+1))
            print('prediction_loss:'+str(float(prediction_loss))+' weight_loss:'+str(float(weight_loss))+' bias_loss:'+str(float(bias_loss))+' feature_loss:'+str(float(feature_loss)))
            total_loss /= len(minibatches_idx)
            print('epoch:', epoch+1, 'loss:', total_loss)
                      
        
def simple_train_batch_warm(trainloader, model, loss_function, optimizer, epochs, lr, batch_size, lambda_s,save_dir,N):
    if os.path.exists(save_dir):
        pass
    else:
        os.mkdir(save_dir)
    model.train()
    for epoch in range(epochs):
        if epoch == 0:
            print('starting with a larger regularization parameter')
            lambda_s[0],lambda_s[1] = lambda_s[0]/10,lambda_s[1]/10
        if epoch == int(epochs / 6):
            print('setting regularization parameter to original level')
            lambda_s[0],lambda_s[1] = lambda_s[0]*10,lambda_s[1]*10
        if epoch == int(epochs / 3):
            for g in optimizer.param_groups:
                g['lr'] = lr / 10
            print('divide current learning rate by 10')
        elif epoch == int(epochs /3*2):
            for g in optimizer.param_groups:
                g['lr'] = lr / 100
            print('divide current learning rate by 10')
        total_loss = 0
        feature_loss = 0
        minibatches_idx = get_minibatches_idx(len(trainloader), minibatch_size=batch_size,
                                              shuffle=True)
        for minibatch in minibatches_idx:
            inputs = torch.Tensor(np.array([list(trainloader[x][0].cpu().numpy()) for x in minibatch]))
            targets = torch.Tensor(np.array([list(trainloader[x][1].cpu().numpy()) for x in minibatch]))
            inputs, targets = Variable(inputs.cuda()).squeeze(1), Variable(targets.long().cuda()).squeeze()
            optimizer.zero_grad()
            outputs, features_f_re = model(inputs)
            outputs = outputs.squeeze()
            weight_loss = lambda_s[1]*torch.sum(model.classifier.weight**2)
            feature_loss =  lambda_s[0]*torch.sum(features_f_re**2)/len(targets)*N
            bias_loss = lambda_s[2]*torch.sum(model.classifier.bias**2)
            prediction_loss = loss_function(outputs, targets)
            loss = prediction_loss + weight_loss + bias_loss + feature_loss
            total_loss += loss
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0 or (epoch+1) <= 5:
            print('prediction_loss:'+str(float(prediction_loss))+' weight_loss:'+str(float(weight_loss))+' bias_loss:'+str(float(bias_loss))+' feature_loss:'+str(float(feature_loss)))
            total_loss /= len(minibatches_idx)
            print('epoch:', epoch+1, 'loss:', total_loss)
    torch.save(model.state_dict(), save_dir+'/network_weight')
            
def simple_train_batch_2(trainloader, model, loss_function, optimizer, epochs, lr, batch_size, lambda_s,save_dir,N):
    if os.path.exists(save_dir):
        pass
    else:
        os.mkdir(save_dir)
    model.train()
    for epoch in range(epochs):
        if epoch == int(epochs / 6):
            for g in optimizer.param_groups:
                g['lr'] = lr / 10
            print('divide current learning rate by 10')
        elif epoch == int(epochs /3):
            for g in optimizer.param_groups:
                g['lr'] = lr / 100
            print('divide current learning rate by 10')
        total_loss = 0
        feature_loss = 0
        minibatches_idx = get_minibatches_idx(len(trainloader), minibatch_size=batch_size,
                                              shuffle=True)
        
        for minibatch in minibatches_idx:
            inputs = torch.Tensor(np.array([list(trainloader[x][0].cpu().numpy()) for x in minibatch]))
            targets = torch.Tensor(np.array([list(trainloader[x][1].cpu().numpy()) for x in minibatch]))
            inputs, targets = Variable(inputs.cuda()).squeeze(1), Variable(targets.long().cuda()).squeeze()
            optimizer.zero_grad()
            outputs, features_f_re = model(inputs)
            outputs = outputs.squeeze()
            weight_loss = lambda_s[1]*torch.sum(model.classifier.weight**2)
            feature_loss =  lambda_s[0]*torch.sum(features_f_re**2)/len(targets)*N
            bias_loss = lambda_s[2]*torch.sum(model.classifier.bias**2)
            prediction_loss = loss_function(outputs, targets)
            loss = prediction_loss + weight_loss + bias_loss + feature_loss
            total_loss += loss
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0 or (epoch+1) <= 5:
            print('prediction_loss:'+str(float(prediction_loss))+' weight_loss:'+str(float(weight_loss))+' bias_loss:'+str(float(bias_loss))+' feature_loss:'+str(float(feature_loss)))
            total_loss /= len(minibatches_idx)
            print('epoch:', epoch+1, 'loss:', total_loss)
    torch.save(model.state_dict(), save_dir+'/network_weight')