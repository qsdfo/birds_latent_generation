import time
import numpy as np
from avgn.pytorch.dataset.spectro_dataset import SpectroDataset
from avgn.signalprocessing.spectrogramming_scipy import build_mel_basis
from avgn.signalprocessing.create_spectrogram_dataset import prepare_wav
from main_spectrogramming import process_syllable
import os
import shutil

import click
from torch.utils.tensorboard import SummaryWriter
from avgn.pytorch.generate.generation import plot_generation
from avgn.pytorch.generate.interpolation import plot_interpolations
from avgn.pytorch.generate.reconstruction import plot_reconstruction
from avgn.pytorch.getters import get_dataloader, get_model_and_dataset
from avgn.pytorch.generate.plot_tsne_latent import plot_tsne_latent


@click.command()
@click.option('-c', '--config', type=str)
@click.option('-l', '--load', type=str)
@click.option('-t', '--train', is_flag=True)
def main(config,
         load,
         train):

    # Init model and dataset
    model, dataset_train, dataset_val, optimizer, config, model_path, config_path = get_model_and_dataset(
        config=config, loading_epoch=load)

    # Training
    best_val_loss = float('inf')
    num_examples_plot = 10
    if train:
        print('##### Train')
        # Copy config file in the save directory before training
        if not load:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
                os.mkdir(f'{model.model_dir}/training_plots/')
                os.mkdir(f'{model.model_dir}/plots/')
            shutil.copy(config_path, f'{model_path}/config.py')

        writer = SummaryWriter(f'{model.model_dir}')

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
            writer.add_scalar('train_loss', train_loss, ind_epoch)
            writer.add_scalar('val_loss', val_loss, ind_epoch)

            print(f'Epoch {ind_epoch}:')
            print(f'Train loss {train_loss}:')
            print(f'Val loss {val_loss}:')

            del train_dataloader, val_dataloader

            # if (val_loss < best_val_loss) and (ind_epoch % 200 == 0):
            if ind_epoch % 10 == 0:
                # Save model
                model.save(name=ind_epoch)

                # Plots
                if not os.path.isdir(f'{model.model_dir}/training_plots/{ind_epoch}'):
                    os.mkdir(f'{model.model_dir}/training_plots/{ind_epoch}')
                test_dataloader = get_dataloader(dataset_type=config['dataset'], dataset=dataset_val,
                                                 batch_size=num_examples_plot, shuffle=True)
                savepath = f'{model.model_dir}/training_plots/{ind_epoch}/reconstructions'
                os.mkdir(savepath)
                plot_reconstruction(model, hparams, test_dataloader, savepath, custom_data=None)
                savepath = f'{model.model_dir}/training_plots/{ind_epoch}/generations'
                os.mkdir(savepath)
                plot_generation(model, hparams, num_examples_plot, savepath)
                del test_dataloader

    # Generations
    print('##### Generate')
    USE_CUSTOM_SAMPLE = False
    # Use own samples for generation
    if USE_CUSTOM_SAMPLE:
        data_source = {
            0: {
                'start': {'path': 'data/raw/source_generation/0.wav'},
                'end': {'path': 'data/raw/source_generation/1.wav'},
            },
            1: {
                'start': {'path': 'data/raw/source_generation/0.wav'},
                'end': {'path': 'data/raw/source_generation/2.wav'},
            },
            2: {
                'start': {'path': 'data/raw/source_generation/1.wav'},
                'end': {'path': 'data/raw/source_generation/2.wav'},
            }
        }
        start_data = []
        end_data = []
        for example_ind, examples_dict in data_source.items():
            for name in ['start', 'end']:
                # read file
                syl, _ = prepare_wav(
                    wav_loc=examples_dict[name]['path'], hparams=hparams, debug=False)
                mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
                # process syllable
                sn, mSp, _ = process_syllable(
                    syl=syl, hparams=hparams, mel_basis=mel_basis, debug=False)
                if name == 'start':
                    start_data.append(SpectroDataset.process_mSp(mSp))
                elif name == 'end':
                    end_data.append(SpectroDataset.process_mSp(mSp))
        all_data = start_data + end_data
        #  Batchify
        start_data = np.stack(start_data)
        end_data = np.stack(end_data)
        all_data = np.stack(all_data)
        custom_data = {
            'start_data': start_data,
            'end_data': end_data,
            'all_data': all_data,
        }
    else:
        custom_data = None

    # Dataloader from the dataset, used for latent space visualisation and to feed data for reconstruction
    # and interpolation if not custom examples are provided
    gen_dataloader = get_dataloader(dataset_type=config['dataset'], dataset=dataset_val,
                                    batch_size=num_examples_plot, shuffle=True)

    if not os.path.isdir(f'{model.model_dir}/plots'):
        os.mkdir(f'{model.model_dir}/plots')

    # Reconstructions
    savepath = f'{model.model_dir}/plots/reconstructions'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    plot_reconstruction(model, hparams, gen_dataloader, savepath, custom_data=custom_data)

    # Sampling
    savepath = f'{model.model_dir}/plots/generations'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    plot_generation(model, hparams, num_examples_plot, savepath)

    # Linear interpolations
    savepath = f'{model.model_dir}/plots/linear_interpolations'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    plot_interpolations(model, hparams, gen_dataloader, savepath, num_interpolated_points=10, method='linear',
                        custom_data=custom_data)

    # Constant r interpolations
    savepath = f'{model.model_dir}/plots/constant_r_interpolations'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    plot_interpolations(model, hparams, gen_dataloader, savepath,
                        num_interpolated_points=10, method='constant_radius', custom_data=custom_data)

    # TODO
    # Translations

    # Check geometric organistation of the latent space per species
    savepath = f'{model.model_dir}/plots/stats'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    # latent_space_stats_per_species(model, gen_dataloader, savepath)
    plot_tsne_latent(model, gen_dataloader, savepath)


def epoch(model, optimizer, dataloader, num_batches, training):
    if training:
        model.train()
    else:
        model.eval()
    losses = []
    for batch_idx, data in enumerate(dataloader):
        aaa = time.time()
        if num_batches is not None and batch_idx > num_batches:
            break
        optimizer.zero_grad()
        bbb = time.time()
        loss = model.step(data)
        ccc = time.time()
        if training:
            loss.backward()
            optimizer.step()
        ddd = time.time()
        print(f'forward {bbb-aaa}, backward {ddd-ccc}')
        losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    return mean_loss


if __name__ == '__main__':
    main()
