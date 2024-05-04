from modules import UNet_conditional
import torch
import pandas as pd
from torchsummary import summary
from utils import*


class_table = torch.Tensor(pd.read_csv("data/class_table.csv", index_col=False).to_numpy().T)
model = UNet_conditional()
t = (torch.ones(1)*500).long()
print(t)
SHP, Loc, CR, SK, HT, FT, Mag = class_maker(1, [1], class_table )
print(class_maker(1, [1], class_table ))
summary(model, [(1,512,512), t, torch.Tensor([0]),torch.Tensor([0]),torch.Tensor([0]),
                torch.Tensor([0]),torch.Tensor([0]),torch.Tensor([0]),torch.Tensor([0])],
                dtypes=[torch.float, torch.long, torch.long, torch.long, torch.long, torch.long, torch.long, torch.long])
