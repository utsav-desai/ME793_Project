import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
import torchvision.utils
from torchvision import datasets, transforms
from torch.distributions import uniform
from mpl_toolkits.axes_grid1 import ImageGrid
import os

ROOT = os.path.dirname(__file__)

def get_config():
    setup_file = os.path.join(ROOT,"config.json")
    with open(setup_file,'r') as file:
        config = json.loads(file.read())
    return config

fig = plt.figure(figsize = (100,100))
def show_images(images, index, label):
    ax = fig.add_subplot(21, 1,index+1, xticks = [], yticks = [])
    plt.gca().set_title(label)
    #ax.set_title(label,fontsize = 40)
    plt.imshow(images.cpu(), cmap = 'gray')
    plt.show()
    

def show_grids(images, labels, n_epoch, label_dict, config):
    labels_title = []
    for i in labels:
        labels_title.append(label_dict[i.item()])
    fig = plt.figure(figsize=(40., 40.))
    grid = ImageGrid(fig, 111, 
                     nrows_ncols=(2, 2),  
                     axes_pad=0.5
                     )
    j = 0
    for ax, im in zip(grid, images.cpu().view(-1,config['img_size'],config['img_size'])):
        ax.imshow(im, cmap = 'gray')
        ax.title.set_text(labels_title[j])
        ax.title.set_size(28)
        #fig.gca().set_title(labelt[i])
        j += 1
    #plt.show()
    figname = str(ROOT) + '/Generated-Images/' + str(1200+n_epoch)
    fig.savefig(figname, bbox_inches='tight')
    plt.close(fig)

def show_images(images, index, label):
    fig = plt.figure(figsize = (100,100))
    ax = fig.add_subplot(21, 1,index+1, xticks = [], yticks = [])
    plt.gca().set_title(label)
    #ax.set_title(label,fontsize = 40)
    plt.imshow(images.cpu(), cmap = 'gray')
    plt.show()
    
## Model Utilities

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('Norm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t * 255.),
    ])


def class_maker(batch_size, labels, class_table):
    SHP = torch.zeros([batch_size])
    Loc = torch.zeros([batch_size])
    CR = torch.zeros([batch_size])
    FT = torch.zeros([batch_size])
    SK = torch.zeros([batch_size])
    HT = torch.zeros([batch_size])
    Mag = torch.zeros([batch_size])
    for i in range(batch_size):
        SHP[i] = class_table[0,labels[i]]
        Loc[i] = class_table[1,labels[i]]
        CR[i] = class_table[2,labels[i]]
        SK[i] = class_table[3,labels[i]]
        HT[i] = class_table[4,labels[i]]
        FT[i] = class_table[5,labels[i]]
        Mag[i] = class_table[6,labels[i]]

    return SHP, Loc, CR, SK, HT, FT, Mag

# def save_model(address):
#     checkpoint = {"model_state": model.state_dict(),
#                    "ema_model_state": ema_model.state_dict(),
#               "model_optimizer": optimizer.state_dict()}
#     torch.save(checkpoint, address)
        
# def load_model(address):
#     if torch.cuda.is_available() == True:
#         checkpoint = torch.load(address)
#         model.load_state_dict(checkpoint["model_state"])
#         ema_model.load_state_dict(checkpoint["ema_model_state"])
#         optimizer.load_state_dict(checkpoint["model_optimizer"])
#     else:
#         checkpoint = torch.load(address, map_location=torch.device('cpu'))
#         model.load_state_dict(checkpoint["model_state"])
#         ema_model.load_state_dict(checkpoint["ema_model_state"])
#         optimizer.load_state_dict(checkpoint["model_optimizer"])

# def load_model2(address):
#     checkpoint = torch.load(address)
#     model.load_state_dict(checkpoint["model_state"])
#     optimizer.load_state_dict(checkpoint["model_optimizer"])