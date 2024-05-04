import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

## CONFIG
ROOT = os.path.dirname(__file__)
setup_file = os.path.join(ROOT,"config.json")
with open(setup_file,'r') as file:
    config = json.loads(file.read())

image_size = config["image_size"]
device = config["device"]
shapes = config["shapes"]
locations = config["locations"]
cooling_rates = config["cooling_rates"]
soaking_times = config["soaking_times"]
forging_temps = config["forging_temps"]
heat_treatments = config["heat_treatments"]
magnifications = config["magnifications"]
embedding_dim = config["embedding_dim"]
num_classes = config["num_classes"]

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=image_size):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:,None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat*x + sqrt_one_minus_alpha_hat*epsilon, epsilon
    
    def sample_timesteps(self,n):
        return torch.randint(low = 1, high = self.noise_steps, size = (n,))
    
    def sample(self, model, n, SHP, Loc, CR, SK, HT, FT, Mag, cfg_scale=0):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(device)
            for i in reversed(range(1, self.noise_steps)):
                print(i)
                t = (torch.ones(n)*i).long().to(device)
                predicted_noise = model(x,t, SHP, Loc, CR, SK, HT, FT, Mag)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x,t, None, None, None, None, None, None, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i>1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        #x = (x.clamp(-1,1)+1)/2
        #x = (x*255).type(torch.uint8)
        return x        
    
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0
        
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
            
    def update_average(self, old, new):
        if old is None:
            return new
        return old*self.beta + (1-self.beta)*new
    
    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1
        
    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
        

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
        
    def forward(self,x):
        x = x.view(-1, self.channels, self.size*self.size).swapaxes(1,2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2,1).view(-1, self.channels, self.size, self.size)
    
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None, residual = False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels,kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels)
        )
    
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
        
        
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, imsize, emb_dim=512):
        super().__init__()
        self.imsize = imsize
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, (out_channels-7))
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )
        self.SHP_label = nn.Sequential(
                                nn.Embedding(shapes, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.Loc_label = nn.Sequential(
                                nn.Embedding(locations, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.CR_label = nn.Sequential(
                                nn.Embedding(cooling_rates, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.SK_label = nn.Sequential(
                                nn.Embedding(soaking_times, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.HT_label = nn.Sequential(
                                nn.Embedding(heat_treatments, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.FT_label = nn.Sequential(
                                nn.Embedding(forging_temps, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.Mag_label = nn.Sequential(
                                nn.Embedding(magnifications, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        
    def forward(self, x, t, SHP, Loc, CR, SK, HT, FT, Mag):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1])
        shpemb = self.SHP_label(SHP).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        locemb = self.Loc_label(Loc).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        cremb = self.CR_label(CR).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        skemb = self.SK_label(SK).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        htemb = self.HT_label(HT).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        ftemb = self.FT_label(FT).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        magemb = self.Mag_label(Mag).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        x = torch.cat((x, shpemb, locemb, cremb, skemb, htemb, ftemb, magemb), dim = 1)
        return x + emb
    
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, imsize, emb_dim=512):
        super().__init__()
        self.imsize = imsize
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels-7, in_channels // 2)
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )
        self.SHP_label = nn.Sequential(
                                nn.Embedding(shapes, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.Loc_label = nn.Sequential(
                                nn.Embedding(locations, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.CR_label = nn.Sequential(
                                nn.Embedding(cooling_rates, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.SK_label = nn.Sequential(
                                nn.Embedding(soaking_times, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.HT_label = nn.Sequential(
                                nn.Embedding(heat_treatments, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.FT_label = nn.Sequential(
                                nn.Embedding(forging_temps, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.Mag_label = nn.Sequential(
                                nn.Embedding(magnifications, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        
    def forward(self, x, skip_x, t, SHP, Loc, CR, SK, HT, FT, Mag):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim = 1)
        x = self.conv(x)
        shpemb = self.SHP_label(SHP).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        locemb = self.Loc_label(Loc).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        cremb = self.CR_label(CR).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        skemb = self.SK_label(SK).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        htemb = self.HT_label(HT).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        ftemb = self.FT_label(FT).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        magemb = self.Mag_label(Mag).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        x = torch.cat((x, shpemb, locemb, cremb, skemb, htemb, ftemb, magemb), dim = 1)
        emb = self.emb_layer(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1])
        return x + emb
    
    

class UNet_conditional(nn.Module):
    def __init__(self,c_in = 1, c_out = 1, time_dim=512, num_classes=None):
        super().__init__()
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 16)
        self.down1 = Down(16,32, 256)
        self.sa1 = SelfAttention(32, 256)
        self.down2 = Down(32, 64, 128)
        self.sa2 = SelfAttention(64, 128)
        self.down3 = Down(64, 128, 64)
        self.sa3 = SelfAttention(128, 64)
        self.down4 = Down(128, 256, 32)
        self.sa4 = SelfAttention(256, 32)
        self.down5 = Down(256, 512, 16)
        self.sa5 = SelfAttention(512, 16)
        self.down6 = Down(512, 512, 8)
        self.sa6 = SelfAttention(512, 8)
        
        self.bot1 = DoubleConv(512, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 512)
        
        self.up6 = Up(1024, 256, 16)
        self.as6 = SelfAttention(256, 16)
        self.up5 = Up(512, 128, 32)
        self.as5 = SelfAttention(128, 32)
        self.up4 = Up(256, 64, 64)
        self.as4 = SelfAttention(64, 64)
        self.up3 = Up(128, 32, 128)
        self.as3 = SelfAttention(32, 128)
        self.up2 = Up(64, 16, 256)
        self.as2 = SelfAttention(16, 256)
        self.up1 = Up(32, 8, 512)
        self.as1 = SelfAttention(8, 512)
        self.outc = nn.Conv2d(8, c_out, kernel_size = 1)
        
        
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2).float().to(device) / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2)*inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2)*inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t, SHP, Loc, CR, SK, HT, FT, Mag):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
            
        
        x0 = self.inc(x)
        x1 = self.down1(x0, t, SHP, Loc, CR, SK, HT, FT, Mag)
        #x1 = self.sa1(x1)
        x2 = self.down2(x1, t, SHP, Loc, CR, SK, HT, FT, Mag)
        #x2 = self.sa2(x2)
        x3 = self.down3(x2, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x3 = self.sa3(x3)
        x4 = self.down4(x3, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x4 = self.sa4(x4)
        x5 = self.down5(x4, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x5 = self.sa5(x5)
        x6 = self.down6(x5, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x6 = self.sa6(x6)
        
        x6 = self.bot1(x6)
        x6 = self.bot2(x6)
        x6 = self.bot3(x6)
        
        x = self.up6(x6, x5, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x = self.as6(x)
        x = self.up5(x, x4, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x = self.as5(x)
        x = self.up4(x, x3, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x = self.as4(x)
        x = self.up3(x, x2, t, SHP, Loc, CR, SK, HT, FT, Mag)
        #x = self.as3(x)
        x = self.up2(x, x1, t, SHP, Loc, CR, SK, HT, FT, Mag)
        #x = self.as2(x)
        x = self.up1(x, x0, t, SHP, Loc, CR, SK, HT, FT, Mag)
        #x = self.as1(x)
        output = self.outc(x)
        return output
