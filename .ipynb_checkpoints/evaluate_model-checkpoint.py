import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from model import Encoder, Decoder, Decoder3
from train import train
from dataset.fusion_generator import Fusion_Dataset, evaluation_augmentations
from dataset.ifs_tc_generator import IFS_TC, evaluation_augmentations
from dataset.columbia_generator import Columbia_Dataset, evaluation_augmentations
from torch import nn
from utils import Binary_Metrics
from training import evaluate
import click
import click

@click.group()
def cli():
    pass

@cli.command()
@click.option('--experiment_name', type=str, default='', show_default=True)
@click.option('--model_name', required=True, type=str, show_default=True)
@click.option('--resume', is_flag=True, show_default=True)
@click.option('--batch_size', required=True, type=int, default=16, show_default=True)
@click.option('--gpu_id', type=str, default='0', show_default=True)
@click.option('--checkpoint_path', type=str, default='checkpoints/', show_default=True)
@click.option('--dataset_name', required=True, show_default=True)
@click.option('--dataset_path', reauired=True, show_default=True)
@click.option('--dataset_csv', required=True, show_default=True)

def evaluating(experiment_name, model_name, batch_size, gpu_id, checkpoint_path, dataset_name, dataset_path, dataset_csv):

    kwargs = locals()
    print(kwargs)

    experiment_name= experiment_name
    writer = SummaryWriter('runs/'+experiment_name)
    
    checkpoint_path = os.path.join(checkpoint_path, experiment_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    if gpu_id is not None:
        device = torch.device('cuda:'+str(gpu_id) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
        
    if model_name == 'dct':
        flag_dct = True
        flag_sb = False
        model = Decoder()
    if model_name == 'sb':
        flag_dct = False
        flag_sb = True
        model = Decoder()
    if model_name == 'dct_sb':
        flag_dct = True
        flag_sb = True
        model = Decoder3()
    
    if dataset_name == 'casia1':
        eval_data = Data_Generator(dataset_path, dataset_csv, transform=evaluate_augmentations, split=['train', 'val'], dct=flag_dct, sb=flag_sb)
    if dataset_name == 'ifs_tc':
        eval_data = Data_Generator(dataset_path, dataset_csv, transform=evaluate_augmentations, split=['test'], dct=flag_dct, sb=flag_sb, inverse=True)
    if dataset_name == 'columbia':
        eval_data = Data_Generator(dataset_path, dataset_csv, transform=evaluate_augmentations, split=['test'], dct=flag_dct, sb=flag_sb)


    eval_loader = DataLoader(dataset = eval_data, shuffle = False, batch_size = batch_size, num_workers=16, pin_memory=True)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model = model.to(device)
    criterion = nn.BCELoss().to(device)

    val_metrics = Binary_Metrics()
    val_iterator = iter(eval_loader)

    val_metrics = evaluate(model, criterion, val_iterator, device, val_metrics, flag_dct, flag_sb)
    print(val_metrics)