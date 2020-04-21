import importlib
import os
from datetime import datetime

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from avgn.pytorch.decoder import Decoder
from avgn.pytorch.encoder import Encoder
from avgn.pytorch.getters import get_model
from avgn.pytorch.spectro_dataset import SpectroDataset, cuda_variable
from avgn.utils.paths import DATA_DIR, MODEL_DIR


@click.command()
@click.option('-p', '--plot', is_flag=True)
@click.option('-d', '--dataset', type=str)
@click.option('-c', '--config', type=str)
@click.option('-l', '--load', type=str)
@click.option('-t', '--train', is_flag=True)
def main(plot,
         dataset,
         config,
         load,
         train):
    # Use all gpus available
    gpu_ids = [int(gpu) for gpu in range(torch.cuda.device_count())]
    print(f'Using GPUs {gpu_ids}')
    if len(gpu_ids) == 0:
        device = 'cpu'
    else:
        device = 'cuda'

    # Load config
    config_path = config
    config_module_name = os.path.splitext(config)[0].replace('/', '.')
    config = importlib.import_module(config_module_name).config

    # compute time stamp
    if config['timestamp'] is not None:
        timestamp = config['timestamp']
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        config['timestamp'] = timestamp

    if load:
        model_dir = os.path.dirname(config_path)
    else:
        model_dir = f'models/{config["savename"]}_{timestamp}'

    ##################################################################################
    print(f'##### Dataset')
    dataset_name = dataset
    df_loc = DATA_DIR / 'syllable_dfs' / dataset_name / 'data.pickle'
    syllable_df = pd.read_pickle(df_loc)
    num_examples = len(syllable_df)
    split = 0.9
    syllable_df_train = syllable_df.iloc[: int(split * num_examples)].reset_index()
    syllable_df_test = syllable_df.iloc[int(split * num_examples):].reset_index()
    dataset_train = SpectroDataset(syllable_df_train)
    dataset_test = SpectroDataset(syllable_df_test)
    example_data = dataset_train[0]
    dims = example_data.shape[1:]

    # Dataloaders
    train_dataloader = DataLoader(dataset_train,
                                  batch_size=config['batch_size'],
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True,
                                  )
    val_dataloader = DataLoader(dataset_test,
                                batch_size=config['batch_size'],
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True,
                                )

    ##################################################################################
    print(f'##### Model')
    encoder = Encoder(input_dim=dims, n_z=config['n_z'])
    decoder = Decoder(deconv_input_shape=(64, 8, 6), n_z=config['n_z'])
    model = get_model(model_type=config['model_type'],
                      model_kwargs=config['model_kwargs'],
                      encoder=encoder,
                      decoder=decoder)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=config['lr'])

    if load is not None:
        print(f'##### Load model')
        loadpath = MODEL_DIR / load
        model.load(path=loadpath, device=device)

    model.to(device)

    ##################################################################################
    print(f'##### Training')
    best_val_loss = float('inf')
    if train:
        for ind_epoch in range(config['num_epochs']):
            train_loss = epoch(model, optimizer, train_dataloader, training=True)
            val_loss = epoch(model, optimizer, val_dataloader, training=False)
            print(f'Epoch {ind_epoch}:')
            print(f'Train loss {train_loss}:')
            print(f'Val loss {val_loss}:')

            if (val_loss < best_val_loss) or (ind_epoch % 10 == 0):
                savedir = MODEL_DIR / f'{config["model_type"]}-{ind_epoch}-{timestamp}'
                if os.path.isdir(savedir):
                    os.rmdir(savedir)
                os.mkdir(savedir)
                model.save(path=savedir)


def epoch(model, optimizer, dataloader, training):
    if training:
        model.train()
    else:
        model.eval()
    losses = []
    for batch_idx, data in enumerate(dataloader):
        data_cuda = cuda_variable(torch.tensor(data))
        optimizer.zero_grad()
        loss = model.step(data_cuda)
        if training:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    return mean_loss


if __name__ == '__main__':
    main()
