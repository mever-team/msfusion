import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from model import Encoder, Decoder, Decoder3
from train import train
from torch import nn
from dataset.synthetic_generator import Synthetic_Dataset, train_augmentations, val_augmentations
from dataset.fusion_generator import Fusion_Dataset, train_augmentations, val_augmentations
from dataset.ifs_tc_generator import IFS_TC, train_augmentations, val_augmentations
from torch.utils.tensorboard import SummaryWriter
import click
import os


@click.group()
def cli():
    pass

@cli.command()
@click.option('--experiment_name', type=str, default='', show_default=True)
@click.option('--model_name', required=True, type=str, show_default=True)
@click.option('--resume', is_flag=True, show_default=True)
@click.option('--batch_size', required=True, type=int, default=16, show_default=True)
@click.option('--gpu_id', type=str, default='0', show_default=True)
@click.option('--learning_rate', type=float, default=1e-3, show_default=True)
@click.option('--epochs', type=int, default=20, show_default=True)
@click.option('--checkpoint_path', type=str, default='checkpoints/', show_default=True)
@click.option('--dataset_name', required=True, show_default=True)
@click.option('--dataset_path', required=True, show_default=True)
@click.option('--dataset_csv', required=True, show_default=True)


def training(experiment_name, model_name, batch_size, gpu_id, learning_rate, epochs, checkpoint_path, dataset_name, dataset_path, dataset_csv):
    
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
    
    if resume:    
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    
    if dataset_name == 'synthetic':
        train_data = Data_Generator(dataset_path, dataset_csv, transform=train_augmentations, dct=flag_dct, sb=flag_sb, inverse=True)
        val_data = Data_Generator(dataset_path, dataset_csv, transform=val_augmentations, split=['val'], dct=flag_dct, sb=flag_sb, inverse=True)
    if dataset_name == 'casia2':
        train_data = Data_Generator(dataset_path, dataset_csv, transform=train_augmentations, dct=flag_dct, sb=flag_sb)
        val_data = Data_Generator(dataset_path, dataset_csv, transform=val_augmentations, split=['val'], dct=flag_dct, sb=flag_sb)
    if dataset_name == 'ifs_tc':
        train_data = Data_Generator(dataset_path, dataset_csv, transform=train_augmentations, dct=flag_dct, sb=flag_sb, inverse=True)
        val_data = Data_Generator(dataset_path, dataset_csv, transform=val_augmentations, split=['val'], dct=flag_dct, sb=flag_sb, inverse=True)

    train_loader = DataLoader(dataset = train_data, shuffle = True, batch_size = batch_size, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset = val_data, shuffle = False, batch_size = batch_size, num_workers=8, pin_memory=True)


    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    train(model, epochs, device, train_loader, val_loader, optimizer, criterion, checkpoint_path=checkpoint_path, writer=writer, dct=flag_dct, sb=flag_sb)
    
if __name__ == '__main__':
    cli()