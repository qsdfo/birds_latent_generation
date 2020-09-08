import glob
import importlib
import os
import pickle
import random
import shutil
from datetime import datetime

import click
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from avgn.pytorch.decoder import Decoder
from avgn.pytorch.encoder import Encoder
from avgn.pytorch.getters import get_model, get_dataloader
from avgn.pytorch.spectro_categorical_dataset import SpectroCategoricalDataset
from avgn.pytorch.spectro_dataset import SpectroDataset
from avgn.signalprocessing.spectrogramming import build_mel_basis, build_mel_inversion_basis, inv_spectrogram_librosa
from avgn.utils.cuda_variable import cuda_variable
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
        hparams = None
    else:
        # Hparams
        hparams_loc = DATA_DIR / 'syllables' / f'{dataset_name}_{config["dataset_preprocessing"]}_hparams.pkl'
        with open(hparams_loc, 'rb') as ff:
            hparams = pickle.load(ff)
        # data
        data_loc = DATA_DIR / 'syllables' / f'{dataset_name}_{config["dataset_preprocessing"]}'
        syllable_paths = glob.glob(f'{str(data_loc)}/[0-9]*')
        random.seed(1234)
        random.shuffle(syllable_paths)
        num_syllables = len(syllable_paths)
        print(f'# {num_syllables} syllables')
        split = 0.9
        syllable_paths_train = syllable_paths[: int(split * num_syllables)]
        syllable_paths_val = syllable_paths[int(split * num_syllables):]
        dataset_train = SpectroDataset(syllable_paths_train)
        dataset_val = SpectroDataset(syllable_paths_val)
        # dataset_train = SpectroCategoricalDataset(syllable_df_train)
        # dataset_val = SpectroCategoricalDataset(syllable_df_val)

    # Get dimensions
    val_dataloader = get_dataloader(dataset_type=config['dataset'],
                                    dataset=dataset_val,
                                    batch_size=3,
                                    shuffle=False)

    # Get image dimensions
    for example_data in val_dataloader:
        dims = example_data['input'].shape[2:]
        if plot:
            fig, ax = plt.subplots(nrows=3)
            for i in range(3):
                # show the image
                ax[i].matshow(example_data['input'][i].reshape(dims), origin="lower")
            plt.show()
            break

    ##################################################################################
    print(f'##### Model')
    encoder_kwargs = config['encoder_kwargs']
    encoder = Encoder(
        n_z=config['n_z'],
        conv_stack=encoder_kwargs['conv_stack'],
        conv2z=encoder_kwargs['conv2z']
    )
    decoder_kwargs = config['decoder_kwargs']
    decoder = Decoder(
        deconv_input_shape=decoder_kwargs['deconv_input_shape'],
        z2deconv=decoder_kwargs['z2deconv'],
        deconv_stack=decoder_kwargs['deconv_stack'],
        n_z=config['n_z']
    )
    model = get_model(
        model_type=config['model_type'],
        model_kwargs=config['model_kwargs'],
        encoder=encoder,
        decoder=decoder,
        model_dir=model_path
    )
    optimizer = torch.optim.Adam(list(model.parameters()), lr=config['lr'])

    if load is not None:
        print(f'##### Load model')
        loadpath = MODEL_DIR / load
        model.load(path=loadpath, device=device)

    model.to(device)

    ##################################################################################
    print(f'##### Training')
    best_val_loss = float('inf')
    num_examples_plot = 10
    if train:
        # Copy config file in the save directory before training
        if not load:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
                os.mkdir(f'{model.model_dir}/training_plots/')
                os.mkdir(f'{model.model_dir}/plots/')
            shutil.copy(config_path, f'{model_path}/config.py')

        # Epochs
        for ind_epoch in range(config['num_epochs']):
            train_dataloader = get_dataloader(dataset_type=config['dataset'], dataset=dataset_train,
                                              batch_size=config['batch_size'], shuffle=True)
            val_dataloader = get_dataloader(dataset_type=config['dataset'], dataset=dataset_val,
                                            batch_size=config['batch_size'], shuffle=True)

            train_loss = epoch(model, optimizer, train_dataloader,
                               num_batches=config['num_batches'], training=True)
            val_loss = epoch(model, optimizer, val_dataloader,
                             num_batches=config['num_batches'], training=False)

            print(f'Epoch {ind_epoch}:')
            print(f'Train loss {train_loss}:')
            print(f'Val loss {val_loss}:')

            del train_dataloader, val_dataloader

            # if (val_loss < best_val_loss) and (ind_epoch % 200 == 0):
            if ind_epoch % 200 == 0:
                # Save model
                model.save(name=ind_epoch)

                # Plots
                if not os.path.isdir(f'{model.model_dir}/training_plots/{ind_epoch}'):
                    os.mkdir(f'{model.model_dir}/training_plots/{ind_epoch}')
                test_dataloader = get_dataloader(dataset_type=config['dataset'], dataset=dataset_val,
                                                 batch_size=num_examples_plot, shuffle=True)
                savepath = f'{model.model_dir}/training_plots/{ind_epoch}/reconstruction'
                plot_reconstruction(model, hparams, test_dataloader, savepath)
                savepath = f'{model.model_dir}/training_plots/{ind_epoch}/generation'
                plot_generation(model, hparams, num_examples_plot, savepath)
                del test_dataloader

    # Generations
    print(f'##### Generating')
    # Reconstructions
    test_dataloader = get_dataloader(dataset_type=config['dataset'], dataset=dataset_val,
                                     batch_size=num_examples_plot, shuffle=True)
    savepath = f'{model.model_dir}/plots/reconstruction'
    plot_reconstruction(model, hparams, test_dataloader, savepath)
    # Sampling
    savepath = f'{model.model_dir}/plots/generation'
    plot_generation(model, hparams, num_examples_plot, savepath)
    # Interpolations
    # Translations


def plot_reconstruction(model, hparams, dataloader, savepath):
    # Forward pass
    model.eval()
    for _, data in enumerate(dataloader):
        x_cuda = cuda_variable(data['input'])
        x_recon = model.reconstruct(x_cuda).cpu().detach().numpy()
        break
    # Plot
    x_orig = data['input'].numpy()
    dims = x_recon.shape[2:]
    num_examples = x_recon.shape[0]
    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=num_examples)
    for i in range(num_examples):
        # show the image
        axes[0, i].matshow(x_orig[i].reshape(dims), origin="lower")
        axes[1, i].matshow(x_recon[i].reshape(dims), origin="lower")
    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(f'{savepath}.pdf')
    plt.close('all')

    # audio
    if hparams is not None:
        mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
        mel_inversion_basis = build_mel_inversion_basis(mel_basis)
        for i in range(num_examples):
            original_audio = inv_spectrogram_librosa(x_orig[i, 0], hparams.sr, hparams,
                                                     mel_inversion_basis=mel_inversion_basis)
            recon_audio = inv_spectrogram_librosa(x_recon[i, 0], hparams.sr, hparams,
                                                  mel_inversion_basis=mel_inversion_basis)
            sf.write(f'{savepath}_{i}_original.wav', original_audio, samplerate=hparams.sr)
            sf.write(f'{savepath}_{i}_recon.wav', recon_audio, samplerate=hparams.sr)


def plot_generation(model, hparams, num_examples, savepath):
    # forward pass
    model.eval()
    gen = model.generate(batch_dim=num_examples).cpu().detach().numpy()

    # plot
    dims = gen.shape[2:]
    plt.clf()
    fig, axes = plt.subplots(ncols=num_examples)
    for i in range(num_examples):
        # show the image
        axes[i].matshow(gen[i].reshape(dims), origin="lower")
    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(f'{savepath}.pdf')
    plt.close('all')

    # audio
    if hparams is not None:
        mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
        mel_inversion_basis = build_mel_inversion_basis(mel_basis)
        for i in range(num_examples):
            gen_audio = inv_spectrogram_librosa(gen[i, 0], hparams.sr, hparams, mel_inversion_basis=mel_inversion_basis)
            sf.write(f'{savepath}_{i}.wav', gen_audio, samplerate=hparams.sr)


def epoch(model, optimizer, dataloader, num_batches, training):
    if training:
        model.train()
    else:
        model.eval()
    losses = []
    for batch_idx, data in enumerate(dataloader):
        if num_batches is not None and batch_idx > num_batches:
            break
        optimizer.zero_grad()
        loss = model.step(data)
        if training:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    return mean_loss


if __name__ == '__main__':
    main()
