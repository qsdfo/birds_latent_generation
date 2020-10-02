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
    model, dataset_train, dataset_val, optimizer, hparams, config, model_path, config_path = get_model_and_dataset(
        config_path=config, loading_epoch=load)

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
            if ind_epoch % 200 == 0:
                # Save model
                model.save(name=ind_epoch)

                # Plots
                if not os.path.isdir(f'{model.model_dir}/training_plots/{ind_epoch}'):
                    os.mkdir(f'{model.model_dir}/training_plots/{ind_epoch}')
                test_dataloader = get_dataloader(dataset_type=config['dataset'], dataset=dataset_val,
                                                 batch_size=num_examples_plot, shuffle=True)
                savepath = f'{model.model_dir}/training_plots/{ind_epoch}/reconstructions'
                os.mkdir(savepath)
                plot_reconstruction(model, hparams, test_dataloader, savepath)
                savepath = f'{model.model_dir}/training_plots/{ind_epoch}/generations'
                os.mkdir(savepath)
                plot_generation(model, hparams, num_examples_plot, savepath)
                del test_dataloader

    # Generations
    print('##### Generate')
    test_dataloader = get_dataloader(dataset_type=config['dataset'], dataset=dataset_val,
                                     batch_size=num_examples_plot, shuffle=True)
    if not os.path.isdir(f'{model.model_dir}/plots'):
        os.mkdir(f'{model.model_dir}/plots')

    # Reconstructions
    # savepath = f'{model.model_dir}/plots/reconstructions'
    # os.mkdir(savepath)
    # plot_reconstruction(model, hparams, test_dataloader, savepath)

    # Sampling
    # savepath = f'{model.model_dir}/plots/generations'
    # os.mkdir(savepath)
    # plot_generation(model, hparams, num_examples_plot, savepath)

    # Linear interpolations
    # savepath = f'{model.model_dir}/plots/linear_interpolations'
    # os.mkdir(savepath)
    # plot_interpolations(model, hparams, test_dataloader, savepath, num_interpolated_points=10, method='linear')

    # Constant r interpolations
    # savepath = f'{model.model_dir}/plots/constant_r_interpolations'
    # os.mkdir(savepath)
    # plot_interpolations(model, hparams, test_dataloader, savepath,
    #                     num_interpolated_points=10, method='constant_radius')

    # TODO
    # Translations

    # Check geometric organistation of the latent space per species
    savepath = f'{model.model_dir}/plots/stats'
    os.mkdir(savepath)
    # latent_space_stats_per_species(model, test_dataloader, savepath)
    plot_tsne_latent(model, test_dataloader, savepath)


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
