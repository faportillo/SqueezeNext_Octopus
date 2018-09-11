from __future__ import print_function
import sys

import tensorflow as tf
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np

import os
import math
import time
from PIL import Image
import random
import train_utils as tu
from logger import Logger

from octo_torch import Octopus


imagenet_path = '/HD1/'
train_path = 'ILSVRC2012_img_train/'
val_path = 'Val_2/'
meta_file = 'Felix/DistillNet/ParallelNet_PyTorch/meta.mat'
config_path = os.getcwd()


def train_network(net, loss_weights=[1.0, 0.3], epochs=1, num_classes=1000, batch_size=100,val_batch_size=50):
    #Load net to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = net.cuda()
    net = net.to(device)
    

    #Load imagenet data using Keras ImageGenerator
    train_data, val_data, train_size, val_size = load_imagenet(num_classes, batch_size, val_batch_size, imagenet_path, train_path, val_path, meta_file, config_path)
    

    steps_per_epoch = train_size//batch_size
    steps_per_val = val_size//val_batch_size
    #val_batch_size=(val_size//val_batch_size)#*val_batch_size

    #Define loss functions
    criterion_main = nn.CrossEntropyLoss()
    criterion_aux = nn.CrossEntropyLoss()

    #Optimizer uses default Adam parameters, same as in Keras model
    optimizer = optim.Adam(net.parameters())

    #Logger for Tensorboard
    logger = Logger('./logs')

    for epoch in range(epochs):

        running_loss = 0.0
        running_acc = 0.0
        '''
            Do Training
        '''
        #Switch to training mode
        net.train()
        train_loss = 0.0
        avg_train_loss = 0.0
        avg_train_acc = 0.0
        count = 0.0
        for i in range(0,train_size-1,batch_size):
            t,l = next(train_data)
            #Convert numpy to torch tensor
            data = torch.from_numpy(t[0]).cuda()
            data = data.permute(0, 3, 1, 2)
            l_main = torch.from_numpy(np.argmax(l[0],axis=1))
            l_aux = torch.from_numpy(np.argmax(l[1],axis=1))
            l_main = l_main.type(torch.LongTensor).cuda()
            l_aux = l_aux.type(torch.LongTensor).cuda()      

            optimizer.zero_grad()
            outputs = net(data)
            
            #Calculate loss and optimize
            loss_main = criterion_main(outputs[0], l_main)
            loss_aux = criterion_aux(outputs[1], l_aux)
            loss = loss_weights[0]*loss_main + loss_weights[1]*loss_aux
            loss.backward()
            optimizer.step()
            
            #Print statistics
            train_acc = accuracy(outputs[0],l_main) 
            print('[%d, %5d] Total Loss: %.5f | Main Loss: %.5f | Aux Loss: %.5f | ACCURACY: %.7f ' %  (epoch+1 , i, loss.item(), loss_main.item(), 0.3*loss_aux.item(), train_acc.item()) )
            train_loss = loss
            avg_train_loss+=train_loss
            avg_train_acc +=train_acc
            count+=1.0
            del t, l , data, l_aux, l_main

        avg_train_loss = avg_train_loss / count
        avg_train_acc = avg_train_acc / count 
        '''
            Do validation
        '''
        #Switch to evaluation mode
        net.eval()
        val_loss = 0.0
        avg_val_loss = 0.0
        avg_val_acc = 0.0
        count=0.0
        for i in range(0,val_size-1,val_batch_size):
            t,l = next(val_data)
            #Convert numpy to torch tensor
            data = torch.from_numpy(t[0]).cuda()
            data = data.permute(0, 3, 1, 2)
            l_main = torch.from_numpy(np.argmax(l[0],axis=1))#.cuda()
            l_aux = torch.from_numpy(np.argmax(l[1],axis=1))#.cuda()  
            l_main = l_main.type(torch.LongTensor).cuda()
            l_aux = l_aux.type(torch.LongTensor).cuda()
            #Calculate outputs
            outputs = net(data)  
            #Calculate loss
            loss_main = criterion_main(outputs[0], l_main)
            loss_aux = criterion_aux(outputs[1], l_aux)
            loss = loss_weights[0]*loss_main + loss_weights[1]*loss_aux
            #Print statistics
            val_acc = accuracy(outputs[0],l_main) 
            val_loss = loss
            avg_val_acc+=val_acc
            avg_val_loss+=loss
            count+=1.0
        
        avg_val_acc = avg_val_acc / count
        avg_val_loss = avg_val_loss / count
        print('[VAL]Total Train Loss: %.5f | Total Train Accuracy: %.5f | Total Val Loss: %.5f | Total Val Accuracy: %.7f ' %  (avg_train_loss, avg_train_acc, avg_val_loss,  avg_val_acc) )
        net.train()
        #Save model and weights
        save_model_training('./train_octopus.pth.tar', epoch, net, optimizer)

        #Log to Tensorboard
        info = {'Training Loss': avg_train_loss.item(), 'Training Accuracy': avg_train_acc.item(), 'Validation Loss': avg_val_loss.item(), 'Validation Accuracy': avg_val_acc.item() }
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch+1)
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
            logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)

    print('Training Complete')
    final_model_path = './saved_models/final_octopus.pth.tar'
    print('Saving final model to '+ final_model_path)
    torch.save(model.state_dict(), final_model_path)
    



def load_imagenet(num_classes, batch_size, val_batch_size, IMAGENET_PATH, TRAIN_PATH, VAL_PATH, META_FILE, CONFIG_PATH):
    start_idx = 0
    end_idx = num_classes-1    
    
    orig_train_img_path = os.path.join(IMAGENET_PATH, TRAIN_PATH)
    orig_val_img_path = os.path.join(IMAGENET_PATH, VAL_PATH)
    train_img_path = orig_train_img_path
    val_img_path = orig_val_img_path
    wnid_labels, _ = tu.load_imagenet_meta(os.path.join(IMAGENET_PATH, \
                                                         META_FILE))
    
    train_img_path = os.path.join(IMAGENET_PATH, 'TrainingClasses__/')
    val_img_path1 = os.path.join(IMAGENET_PATH, 'ValidationClasses__/')
    val_img_path = os.path.join(val_img_path1, str(start_idx)+"_"+str(end_idx))
    tu.create_selective_symbolic_link(start_idx, end_idx, wnid_labels, \
                                      original_training_path=orig_train_img_path, \
                                      new_training_path=train_img_path, \
                                      original_validation_path=orig_val_img_path, \
                                      new_validation_path=val_img_path, \
                                      config_path=CONFIG_PATH)

    train_data, val_data, train_size, val_size = tu.imagenet_generator_multi(train_img_path, \
                                                       val_img_path, batch_size=batch_size, \
                                                       do_augment=True, val_batch_size=val_batch_size)

    return train_data, val_data, train_size, val_size

def save_model_training(filepath, epoch, model, optimizer):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    print("Saving model to " + filepath)
    torch.save(state, filepath)

def load_model_training(filepath):
    state = torch.load(filepath)
    
    return model, optimizer, epoch


def accuracy(out, labels):
    _, argmax = torch.max(out,1)
    accuracy = (labels == argmax.squeeze()).float().mean()
    return accuracy

#Initialize network
octo = Octopus(num_classes=1000)
train_network(octo,num_classes=1000, epochs=1000, batch_size=10,val_batch_size=10)









