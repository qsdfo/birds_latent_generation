import importlib
import os
import shutil
from datetime import datetime

import click
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision import datasets, transforms

from avgn.pytorch.decoder import Decoder
from avgn.pytorch.encoder import Encoder
from avgn.pytorch.getters import get_model, get_dataloader
from avgn.pytorch.spectro_dataset import SpectroDataset
from avgn.utils.paths import DATA_DIR, MODEL_DIR


@click.command()
@click.option('-p', '--plot', is_flag=True)
@click.option('-c', '--config', type=str)
@click.option('-l', '--load', type=str)
@click.option('-t', '--train', is_flag=True)
def main(plot,
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
        model_path = os.path.dirname(config_path)
    else:
        model_path = f'models/{config["savename"]}_{timestamp}'

    ##################################################################################
    print(f'##### Dataset')
    dataset_name = config['dataset']

    if dataset_name == 'mnist':
        dataset_train = datasets.MNIST('data', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ]))
        dataset_val = datasets.MNIST('data', train=False,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))
                                     ]))
    else:
        df_loc = DATA_DIR / 'syllable_dfs' / dataset_name / 'data.pickle'
        syllable_df = pd.read_pickle(df_loc)
        num_examples = len(syllable_df)
        split = 0.9
        syllable_df_train = syllable_df.iloc[: int(split * num_examples)].reset_index()
        syllable_df_val = syllable_df.iloc[int(split * num_examples):].reset_index()
        dataset_train = SpectroDataset(syllable_df_train)
        dataset_val = SpectroDataset(syllable_df_val)

    # Get dimensions
    val_dataloader = get_dataloader(dataset_type=config['dataset'],
                                    dataset=dataset_val,
                                    batch_size=3,
                                    shuffle=False)

    # Get image dimensions
    for example_data in val_dataloader:
        dims = example_data.shape[2:]
        if plot:
            fig, ax = plt.subplots(nrows=3)
            for i in range(3):
                # show the image
                ax[i].matshow(example_data[i].reshape(dims), origin="lower")
            plt.show()
            break

    ##################################################################################
    print(f'##### Model')
    encoder = Encoder(input_dim=dims, n_z=config['n_z'])
    decoder_kwargs = config['decoder_kwargs']
    decoder = Decoder(deconv_input_shape=decoder_kwargs['deconv_input_shape'],
                      deconv_stack=decoder_kwargs['deconv_stack'],
                      n_z=config['n_z'])
    model = get_model(model_type=config['model_type'],
                      model_kwargs=config['model_kwargs'],
                      encoder=encoder,
                      decoder=decoder,
                      model_dir=model_path)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=config['lr'])

    if load is not None:
        print(f'##### Load model')
        loadpath = MODEL_DIR / load
        model.load(path=loadpath, device=device)

    model.to(device)

    ##################################################################################
    print(f'##### Training')
    best_val_loss = float('inf')
    num_examples_plot = 5
    if train:
        # Copy config file in the save directory before training
        if not load:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
                os.mkdir(f'{model.model_dir}/plots/')
            shutil.copy(config_path, f'{model_path}/config.py')

        # Epochs
        for ind_epoch in range(config['num_epochs']):
            train_dataloader = get_dataloader(dataset_type=config['dataset'], dataset=dataset_train,
                                              batch_size=config['batch_size'], shuffle=True)
            val_dataloader = get_dataloader(dataset_type=config['dataset'], dataset=dataset_val,
                                            batch_size=config['batch_size'], shuffle=True)

            train_loss = epoch(model, optimizer, train_dataloader,
                               num_batches=config['num_batches'], training=True, device=device)
            val_loss = epoch(model, optimizer, val_dataloader,
                             num_batches=config['num_batches'], training=False, device=device)

            print(f'Epoch {ind_epoch}:')
            print(f'Train loss {train_loss}:')
            print(f'Val loss {val_loss}:')

            del train_dataloader, val_dataloader

            if (val_loss < best_val_loss) or (ind_epoch % 10 == 0):
                # Save model
                model.save(name=ind_epoch)

                # Plots
                test_dataloader = get_dataloader(dataset_type=config['dataset'], dataset=dataset_val,
                                                 batch_size=num_examples_plot, shuffle=False)
                savepath = f'{model.model_dir}/plots/reconstruction_{ind_epoch}.pdf'
                plot_reconstruction(model, test_dataloader, device, savepath)

                savepath = f'{model.model_dir}/plots/generations_{ind_epoch}.pdf'
                plot_generation(model, num_examples_plot, device, savepath)

                del test_dataloader


def plot_reconstruction(model, dataloader, device, savepath):
    # Forward pass
    model.eval()
    for _, data in enumerate(dataloader):
        data_cuda = data.to(device)
        x_recon = model.reconstruct(data_cuda).cpu().detach().numpy()
        break
    # Plot
    dims = x_recon.shape[2:]
    num_examples = x_recon.shape[0]
    fig, axes = plt.subplots(nrows=2, ncols=num_examples)
    for i in range(num_examples):
        # show the image
        axes[0, i].matshow(data[i].reshape(dims), origin="lower")
        axes[1, i].matshow(x_recon[i].reshape(dims), origin="lower")
    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(savepath)
    plt.clf()


def plot_generation(model, num_examples, device, savepath):
    # forward pass
    model.eval()
    gen = model.generate(batch_dim=num_examples).cpu().detach().numpy()
    # plot
    dims = gen.shape[2:]
    fig, axes = plt.subplots(ncols=num_examples)
    for i in range(num_examples):
        # show the image
        axes[i].matshow(gen[i].reshape(dims), origin="lower")
    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(savepath)
    plt.clf()


def epoch(model, optimizer, dataloader, num_batches, training, device):
    if training:
        model.train()
    else:
        model.eval()
    losses = []
    for batch_idx, data in enumerate(dataloader):
        if num_batches is not None and batch_idx > num_batches:
            break
        data_cuda = data.to(device)
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
