from modules import UNet_conditional
import torch
from torchsummary import summary


model = UNet_conditional()
t = (torch.ones(1)*500).long()
print(t)
summary(model, [(1,512,512), t, torch.Tensor(0),torch.Tensor(0),torch.Tensor(0),torch.Tensor(0),torch.Tensor(0),torch.Tensor(0),torch.Tensor(0)])