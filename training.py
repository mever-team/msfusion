import torch
import numpy as np
from typing import Iterable
from glob import glob
from torch import nn
from tqdm import tqdm
import os
import gc
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from utils import Metrics

def train(model, epochs, device, train_loader, val_loader, optimizer, criterion, checkpoint_path='checkpoints/', step=0, writer=None, monitor_value=None, early_stop_patience=5, dct=False, sb=False):
    model = model.to(device)
    criterion = criterion.to(device)

    val_metrics = Metrics()
    early_stop_patience_counter = early_stop_patience
    
    for epoch in range(epochs):
        for i, data in enumerate(tqdm(train_loader)):
            
            if(dct and sb):
                image, mask, d, s= data
            
            if(dct and not sb):
                image, mask, d = data
                
            if(sb and not dct):
                image, mask, s = data
                
            image = image.to(device)
            mask = mask.to(device)
            
            if dct:
                d = d.to(device)
            if sb:
                s = s.to(device)
                
            model.train()
            optimizer.zero_grad()
            
            if(dct and sb):
                output = model(image, d, s)
            if(dct and not sb):
                output = model(image, d)
            if(sb and not dct):
                output =model(image, s)
                
            loss = criterion(output.float(), mask.float())

            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                writer.add_scalar('train/loss', loss, step)
                
            step += 1
                
        with torch.no_grad():
            
            val_iterator = iter(val_loader)
            val_metric_results = evaluate(model, criterion, val_iterator, device, val_metrics, dct, sb)
            val_loss = val_metric_results['loss']
            
            
            if(monitor_value is None or monitor_value>val_loss):
                monitor_value = val_loss
                early_stop_patience_counter = early_stop_patience
                print('saving to ckpt_{}_{}.pth'.format(epoch,step))
                torch.save({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'loss':val_loss,
                            },os.path.join(checkpoint_path,'ckpt_{}_{}.pth'.format(epoch,step)))
            else:
                early_stop_patience_counter-=1
                if early_stop_patience_counter == 0:
                    print('early stopping')

            
            for metric in val_metric_results:
                if np.size(val_metric_results[metric])>1:
                    for i in range(len(val_metric_results[metric])):
                        writer.add_scalar('val/'+metric+'_'+str(i), val_metric_results[metric][i], epoch)   
                    writer.add_scalar('val/'+metric, np.mean(val_metric_results[metric]), epoch) 
                else:
                    writer.add_scalar('val/'+metric, val_metric_results[metric], epoch)

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
            outputs = model(samples, d, s)
        if(dct and not sb):
            outputs = model(samples, d)
        if(sb and not dct):
            outputs =model(samples, s)
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