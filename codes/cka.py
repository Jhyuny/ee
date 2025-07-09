import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, DistilBertForSequenceClassification,
                          AutoModelForSequenceClassification, get_scheduler,
                          DataCollatorWithPadding)
from datasets import load_dataset
from evaluate import load
from torch.optim import AdamW
from torch import masked_select
from tqdm import tqdm


class CKALoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, SH, TH):
        dT = TH.size(-1)
        dS = SH.size(-1)
        SH = SH.view(-1, dS).to(SH.device, torch.float64)
        TH = TH.view(-1, dT).to(TH.device, torch.float64)
        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)
        num = torch.norm(SH.t().matmul(TH), 'fro')
        den1 = torch.norm(SH.t().matmul(SH), 'fro') + self.eps
        den2 = torch.norm(TH.t().matmul(TH), 'fro') + self.eps
        return 1 - num / torch.sqrt(den1 * den2)

class CKASimilarity(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, SH, TH):
        dT = TH.size(-1)
        dS = SH.size(-1)
        SH = SH.view(-1, dS).to(SH.device, torch.float64)
        TH = TH.view(-1, dT).to(TH.device, torch.float64)
        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)
        num = torch.norm(SH.t().matmul(TH), 'fro')
        den1 = torch.norm(SH.t().matmul(SH), 'fro') + self.eps
        den2 = torch.norm(TH.t().matmul(TH), 'fro') + self.eps
        return num / torch.sqrt(den1 * den2)