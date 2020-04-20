import importlib
import os
from datetime import datetime

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from avgn.pytorch.encoder import Encoder
from avgn.pytorch.spectro_dataset import SpectroDataset
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
    syllable_df_train = syllable_df.iloc[: int(split * num_examples)]
    syllable_df_test = syllable_df.iloc[int(split * num_examples):]
    dataset_train = SpectroDataset(syllable_df_train)
    dataset_test = SpectroDataset(syllable_df_test)
    example_data = dataset_train[0]
    dims = example_data.shape

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
    decoder = Decoder(config['decoder_kwargs'])
    model = get_model(model_type=config['model_type'],
                      model_kwargs=config['model_kwargs'],
                      encoder=encoder,
                      decoder=decoder)

    # encoder = [
    #     tf.keras.layers.Conv2D(
    #         filters=32, kernel_size=3, strides=(2, 2), activation="relu", input_shape=DIMS,
    #     ),
    #     tf.keras.layers.Conv2D(
    #         filters=64, kernel_size=3, strides=(2, 2), activation="relu"
    #     ),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(units=N_Z * 2),
    # ]
    #
    # # Need special deconv stack with weird rectangular spectro... perhaps change that later
    # decoder = [
    #     tf.keras.layers.Dense(units=8 * 6 * 64, activation="relu"),
    #     tf.keras.layers.Reshape(target_shape=(8, 6, 64)),
    #     tf.keras.layers.Conv2DTranspose(
    #         filters=64, kernel_size=3, strides=(4, 2), padding="SAME", activation="relu"
    #     ),
    #     tf.keras.layers.Conv2DTranspose(
    #         filters=32, kernel_size=3, strides=(4, 2), padding="SAME", activation="relu"
    #     ),
    #     tf.keras.layers.Conv2DTranspose(
    #         filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
    #     ),
    # ]

    # model
    if MODEL_TYPE == 'VAE':
        # the optimizer for the model
        optimizer = tf.keras.optimizers.Adam(1e-3)

        # model
        model = VAE(
            enc=encoder,
            dec=decoder,
            optimizer=optimizer
        )

        # a pandas dataframe to save the loss information to
        losses = pd.DataFrame(columns=['recon_loss', 'latent_loss'])

    ##################################################################################
    print(f'##### Model Summary')
    # for spec, index, indv in iter(dataset):
    #     model.train_net(x=spec)

    train_dataset = dataset
    # Todo
    #    create a real test set
    test_dataset = dataset

    # exampled data for plotting results and displaying summary
    example_data, _, _ = next(iter(train_dataset))
    example_data_reshaped = tf.cast(tf.reshape(example_data, (-1, *DIMS)), "float32") / 255.
    model.train_net(example_data_reshaped)

    print(f'# Encoder')
    model.enc.summary()
    print(f'# Decoder')
    model.dec.summary()

    if load is not None:
        print(f'##### Load model')
        loadpath = MODEL_DIR / load
        model.load_model(path=loadpath)

    ##################################################################################
    print(f'##### Training')
    if train:
        n_epochs = 50
        epoch_save = 10
        # inputs_set = False
        for epoch in range(n_epochs):
            # train
            for batch, train_x in zip(range(num_train_batch), train_dataset):
                data_train, ind, indv = train_x
                data_train_proc = tf.cast(tf.reshape(data_train, (-1, *DIMS)), "float32") / 255.
                # if not inputs_set:
                #     model._set_inputs(data_train_proc)
                #     inputs_set = True
                model.train_net(data_train_proc)
            # test on holdout
            loss = []
            for batch, test_x in zip(range(num_test_batches), test_dataset):
                data_test, ind, indv = test_x
                data_test_proc = tf.cast(tf.reshape(data_test, (-1, *DIMS)), "float32") / 255.
                loss.append(model.compute_loss(data_test_proc))
            current_index_loss = len(losses)
            losses.loc[current_index_loss] = np.mean(loss, axis=0)
            # plot results
            print(f"Epoch: {epoch}")
            for loss_name, loss_value in losses.items():
                print(f'{loss_name}: {loss_value[current_index_loss]}')
            if plot:
                if MODEL_TYPE == 'GAIA':
                    model.plot_reconstruction(example_data)
                elif MODEL_TYPE == 'VAE':
                    model.plot_reconstruction(example_data, N_Z)

            # Save model
            if (epoch % epoch_save == 0) or (epoch == n_epochs):
                savepath = MODEL_DIR / f'{MODEL_TYPE}-{epoch}-{TIMESTAMP}'
                if os.path.isdir(savepath):
                    os.rmdir(savepath)
                os.mkdir(savepath)
                model.save_model(savepath)


if __name__ == '__main__':
    main()
