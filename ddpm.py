## Load Libraries
import os
import copy
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from numpy.random import randn
import torchvision.utils
from torch.distributions import uniform
from mpl_toolkits.axes_grid1 import ImageGrid

from utils import *
from modules import *
# from inference import *


## GET TRAINING CONFIG
config = get_config()

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT,"data")
CHECKPOINT_PATH = os.path.join( ROOT , 'checkpoints', 'model.pth.tar')

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = config["batch_size"]
n_epoch = config["epochs"]
n_ax = int(n_epoch/20) 
#n_ax = 1
image_size = config["image_size"]
image_shape = (1,image_size,image_size)
image_dim = int(np.prod(image_shape))
learning_rate = config['lr']
total_loss_min = np.Inf


## LEVELS IN EACH PARAMETER
n_sampled_images = 4
shapes = 2
locations = 3
cooling_rates = 3
soaking_times = 3
forging_temps = 3
heat_treatments = 2
magnifications = 6
embedding_dim = 100
num_classes = 114
# print('Variables Created in ddpm.py')

## Load Class Table
class_table = torch.Tensor(pd.read_csv(os.path.join(ROOT, "data", "class_table.csv"), index_col=False).to_numpy().T)

label_dict = {
    0: 'CM01-0500', 1: 'CM01-1000',2: 'CM01-1500',3: 'CM01-2000',4: 'CM04-0100',5: 'CM04-0500',
    6: 'CM04-1000',7: 'CM04-1500',8: 'CM04-2000',9: 'CM04-3000',10: 'CM10-0500',
    11: 'CM10-1000',12: 'CM10-1500',13: 'CM10-2000',14: 'CM13-0500',15: 'CM13-1000',
    16: 'CM13-2000',17: 'CM13-3000',18: 'CM16-0500',19: 'CM16-1000',20: 'CM16-1500',
    21: 'CM16-2000',22: 'CM16-3000',23: 'CS01-0100',24: 'CS01-0500',25: 'CS01-1000',
    26: 'CS01-1500',27: 'CS01-2000',28: 'CS04-0500',29: 'CS04-1500',30: 'CS04-3000',
    31: 'CS10-0100',32: 'CS10-0500',33: 'CS10-1000',34: 'CS10-1500',35: 'CS10-3000',
    36: 'CS13-0100',37: 'CS13-0500',38: 'CS13-1000',39: 'CS13-2000',40: 'CS13-3000',
    41: 'CS16-0500',42: 'CS16-1000',43: 'CS16-2000',44: 'PL03-0100',45: 'PL03-0500',
    46: 'PL03-1000',47: 'PL03-1500',48: 'PL03-2000',49: 'PL03-3000',50: 'PL11-0100',
    51: 'PL11-0500',52: 'PL11-1000',53: 'PL11-1500',54: 'PL11-2000',55: 'PL11-3000',
    56: 'PL13-0100',57: 'PL13-0500',58: 'PL13-1000',59: 'PL13-2000',60: 'PL13-3000',
    61: 'PL16-0100',62: 'PL16-0500',63: 'PL16-1000',64: 'PL16-1500',65: 'PL16-2000',
    66: 'PL16-3000',67: 'PL18-0100',68: 'PL18-0500',69: 'PL18-1000',70: 'PL18-1500',
    71: 'PL18-2000',72: 'PM03-0100',73: 'PM03-0500',74: 'PM03-1000',75: 'PM03-2000',
    76: 'PM03-3000',77: 'PM11-0100',78: 'PM11-0500',79: 'PM11-1000',80: 'PM11-2000',
    81: 'PM13-0100',82: 'PM13-0500',83: 'PM13-1000',84: 'PM13-2000',85: 'PM13-3000',
    86: 'PM16-0100',87: 'PM16-0500',88: 'PM16-1000',89: 'PM16-2000',90: 'PM16-3000',
    91: 'PM18-0500',92: 'PM18-1000',93: 'PM18-1500',94: 'PM18-2000',95: 'PM18-3000',
    96: 'PS03-0500',97: 'PS03-1000',98: 'PS03-1500',99: 'PS03-2000',100: 'PS03-3000',
    101: 'PS11-0100',102: 'PS11-0500',103: 'PS11-1000',104: 'PS11-2000',105: 'PS16-0100',
    106: 'PS16-0500',107: 'PS16-1000',108: 'PS16-2000',109: 'PS16-3000',110: 'PS18-0100',
    111: 'PS18-0500',112: 'PS18-1000',113: 'PS18-2000'
}

## Load data and create train Loader
train_dataset = datasets.ImageFolder(DATA_DIR, transform = whole_transform)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

print('Making U-Net conditional model')
## Crete Conditional Unet
model = UNet_conditional(num_classes = num_classes).to(device)
print('Model Created')
#model.apply(weights_init)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
print('Optimizer Created')
mse = nn.MSELoss()
diffusion = Diffusion(img_size = image_size)
print('Diffusion Created')
l = len(train_loader)
ema = EMA(0.995)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)

def save_model(address):
    checkpoint = {"model_state": model.state_dict(),
                   "ema_model_state": ema_model.state_dict(),
              "model_optimizer": optimizer.state_dict()}
    torch.save(checkpoint, address)

print('Starting Training')
for e in range(1, n_epoch+1):
    loss_epoch = 0
    for i , (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images = aug_transform(images)
        images = images.to(device)
        shp, loc, cr, sk, ht, ft, mag = class_maker(batch_size = labels.size(0), labels = labels, class_table = class_table)
        shp = shp.long().to(device)
        loc = loc.long().to(device)
        cr = cr.long().to(device)
        sk = sk.long().to(device)
        ht = ht.long().to(device)
        ft = ft.long().to(device)
        mag = mag.long().to(device)
        labels = labels.long().to(device)
        t = diffusion.sample_timesteps(images.shape[0]).to(device)
        x_t, noise = diffusion.noise_images(images, t)
        #if np.random.random() < 0.1:
        #    labels = None
        #    shp = None
        #    loc = None
        #    cr = None
        #    sk = None
        #    ht = None
        #    ft = None
        #    mag = None
        predicted_noise = model(x_t, t, shp, loc, cr, sk, ht, ft, mag)
        loss = mse(noise, predicted_noise)
        
        loss.backward()
        optimizer.step()
        ema.step_ema(ema_model, model)
        loss_epoch += loss.item()
    
    print('Epoch: [%d/%d]: Loss: %.3f' %(
    (e), n_epoch, loss_epoch))
    
    if e % n_ax == 0:
        test_labels = torch.randint(0,114,[n_sampled_images,1])
        t_shp, t_loc, t_cr, t_sk, t_ht, t_ft, t_mag = class_maker(batch_size = test_labels.size(0), labels = test_labels, class_table = class_table)
        test_labels = test_labels.to(device)
        test_labels = test_labels.unsqueeze(1).long()
        t_shp = t_shp.long().to(device)
        t_loc = t_loc.long().to(device)
        t_cr = t_cr.long().to(device)
        t_sk = t_sk.long().to(device)
        t_ht = t_ht.long().to(device)
        t_ft = t_ft.long().to(device)
        t_mag = t_mag.long().to(device)
        #sampled_images = diffusion.sample(model, n_sampled_images, t_shp, t_loc, t_cr, t_sk, t_ht, t_ft, t_mag, cfg_scale=0)
        #sampled_images = reverse_transforms(sampled_images)
        ema_sampled_images = diffusion.sample(ema_model, n_sampled_images, t_shp, t_loc, t_cr, t_sk, t_ht, t_ft, t_mag, cfg_scale=0)
        ema_sampled_images = reverse_transforms(ema_sampled_images)
        # show_grids(ema_sampled_images, test_labels,  e, label_dict, config)
        save_model(CHECKPOINT_PATH)

print(torch.cuda.memory_summary(device=device))
