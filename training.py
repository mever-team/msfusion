import torch
import numpy as np
from typing import Iterable
from utils import Binary_Metrics
from glob import glob
from torch import nn
from tqdm import tqdm
import os
import gc


@torch.no_grad()
def evaluate(model, criterion, iterator, device, metrics, dct=False, sb=False):
    model.eval()
    criterion.eval()
    losses = []
    pbar = tqdm(range(len(iterator)) ,unit='iterations',desc='Validation',postfix={metric:np.nan for metric in metrics.get_metric_names()})
    for iteration in pbar:
        batch = next(iterator)
        
        if(dct and sb):
            samples, targets, d, s= batch
            
        if(dct and not sb):
            samples, targets, d = batch

        if(sb and not dct):
            samples, targets, s = batch
            
        if isinstance(samples,list):
            samples = [s.to(device) for s in samples]
        else:
            samples = samples.to(device)
        targets = targets.to(device)
        
        if dct:
            d = d.to(device)
        if sb:
            s = s.to(device)

        if(dct and sb):
            outputs = model(image, d, s)
        if(dct and not sb):
            outputs = model(image, d)
        if(sb and not dct):
            outputs =model(image, s)
        outputs = torch.squeeze(outputs , 1)
        
        loss = criterion(outputs, targets.float())
        
        losses.append(loss.detach().cpu().numpy())
        targets = targets.long()
        results = metrics.update(outputs.detach().cpu(), targets.detach().cpu())
        results['loss'] = loss.item()
        pbar.set_postfix(**results)
    
    results = metrics.compute()
    results['loss'] = np.mean(losses)
    pbar.set_postfix(**results)
    return results