"""Exact BiContrastiveLoss class copied from WHEN; see source_when.py."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class BiContrastiveLoss(nn.Module):
    def __init__(self, initial_temp=1, final_temp=0.07, total_batches=5):
        super(BiContrastiveLoss, self).__init__()
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_batches = total_batches
        self.current_temp = initial_temp
        self.decay_rate = self.calculate_decay_rate()
 
    def forward(self, clean_feat, adv_feat, bicos, single=False):
        B, D, T = clean_feat.shape
        # print(f"feature shape: {clean_feat.shape}")
        if bicos == 1:
            clean_feat = clean_feat.reshape(B, D, T)
            adv_feat = adv_feat.reshape(B, D, T)
            clean_feat = F.normalize(clean_feat, dim=-1)
            adv_feat = F.normalize(adv_feat, dim=-1)
    
            cos_sim = torch.bmm(clean_feat, adv_feat.transpose(1,2)) / self.current_temp
            labels = torch.arange(cos_sim.size(1), device=cos_sim.device).unsqueeze(0).expand(cos_sim.size(0), -1)

            loss = F.cross_entropy(
            cos_sim.reshape(-1, cos_sim.size(-1)),  
            labels.reshape(-1),
            reduction='none'
        ).reshape(cos_sim.size(0), cos_sim.size(1)).mean(dim=1)
    
        return torch.mean(loss)

    def update_temperature(self, batch_count):
        self.current_temp = self.initial_temp * np.exp(-self.decay_rate * batch_count)
        if self.current_temp <= 0.02:
            self.current_temp = 0.02

    def calculate_decay_rate(self):
        return -np.log(self.final_temp / self.initial_temp) / self.total_batches    #0.23
