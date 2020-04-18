import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_probability.python.distributions import Chi2
from tqdm import tqdm

import avgn.tensorflow.data as tfdata
from avgn.tensorflow.GAIA6 import GAIA, unet_mnist
from avgn.tensorflow.VAE import VAE
from avgn.tensorflow.VAE2 import VAE2
from avgn.utils.paths import DATA_DIR, ensure_dir


@click.command()
@click.option('-p', '--plot', is_flag=True)
@click.option('-d', '--dataset', type=str)
@click.option('-m', '--model', type=str)
def main(plot,
         dataset,
         model):
    ##################################################################################
    # Parameters
    DATASET_ID = dataset
    MODEL_TYPE = model
    N_Z = 32
    TRAIN_BUF = 60000
    BATCH_SIZE = 64
    TEST_BUF = 10000
    N_TRAIN_BATCHES = int(TRAIN_BUF / BATCH_SIZE)
    N_TEST_BATCHES = int(TEST_BUF / BATCH_SIZE)

    ##################################################################################
    print(f'##### Build training dataset')
    df_loc = DATA_DIR / 'syllable_dfs' / DATASET_ID / 'data.pickle'
    syllable_df = pd.read_pickle(df_loc)
    print(f'Dataset contained {len(syllable_df)} syllables')
    ensure_dir(DATA_DIR / 'tfrecords')
    # Load exemple data
    for idx, row in tqdm(syllable_df.iterrows(), total=len(syllable_df)):
        break
    if plot:
        print(f'Spectrogram shape: {np.shape(row.spectrogram)}')

    # Create a tf_records which stores
    record_loc = DATA_DIR / 'tfrecords' / f"{DATASET_ID}.tfrecords"
    with tf.io.TFRecordWriter((record_loc).as_posix()) as writer:
        for idx, row in tqdm(syllable_df.iterrows(), total=len(syllable_df)):
            # FIXME
            data = row.spectrogram[:24, :24]
            DIMS = (*data.shape, 1)
            example = tfdata.serialize_example(
                {
                    "spectrogram": {
                        "data": data.flatten().tobytes(),
                        "_type": tfdata._bytes_feature,
                    },
                    "index": {
                        "data": idx,
                        "_type": tfdata._int64_feature,
                    },
                    "indv": {
                        "data": np.string_(row.indv).astype("|S7"),
                        "_type": tfdata._bytes_feature,
                    },
                }
            )
            # write the defined example into the dataset
            writer.write(example)

    # read the dataset
    raw_dataset = tf.data.TFRecordDataset([record_loc.as_posix()])
    data_types = {
        "spectrogram": tf.uint8,
        "index": tf.int64,
        "indv": tf.string,
    }
    # parse each data type to the raw dataset
    dataset = raw_dataset.map(lambda x: tfdata._parse_function(x, data_types=data_types))
    # shuffle the dataset
    dataset = dataset.shuffle(buffer_size=10000)
    # create batches
    dataset = dataset.batch(10)
    if plot:
        spec, index, indv = next(iter(dataset))
        fig, ax = plt.subplots(ncols=5, figsize=(15, 3))
        for i in range(5):
            # show the image
            ax[i].matshow(spec[i].numpy().reshape(DIMS[:2]), cmap=plt.cm.Greys, origin="lower")
            string_label = indv[i].numpy().decode("utf-8")
            ax[i].set_title(string_label)
            ax[i].axis('off')
        plt.show()

    ##################################################################################
    print(f'##### Model')
    encoder = [
        tf.keras.layers.InputLayer(input_shape=DIMS),
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=(2, 2), activation="relu"
        ),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation="relu"
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=N_Z * 2),
    ]

    # decoder = [
    #     tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu"),
    #     tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
    #     tf.keras.layers.Conv2DTranspose(
    #         filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    #     ),
    #     tf.keras.layers.Conv2DTranspose(
    #         filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    #     ),
    #     tf.keras.layers.Conv2DTranspose(
    #         filters=1, output_shape=DIMS, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
    #     ),
    # ]

    # Need special deconv stack with weird rectangular spectro... perhaps change that later
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

    decoder = [
        tf.keras.layers.Dense(units=6 * 6 * 64, activation="relu"),
        tf.keras.layers.Reshape(target_shape=(6, 6, 64)),
        tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
        ),
        tf.keras.layers.Conv2DTranspose(
            filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
        ),
        tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
        ),
    ]

    # model
    if MODEL_TYPE == 'GAIA':
        # the unet function
        gen_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.5)
        disc_optimizer = tf.keras.optimizers.RMSprop(1e-3)

        model = GAIA(
            dims=DIMS,
            enc=encoder,
            dec=decoder,
            unet_function=unet_mnist,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer,
            chsq=Chi2(df=1 / BATCH_SIZE)
        )

        # a pandas dataframe to save the loss information to
        losses = pd.DataFrame(columns=['d_xg_loss', 'd_xi_loss', 'd_x_loss', 'xg_loss'])

    elif MODEL_TYPE == 'VAE':
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

    elif MODEL_TYPE == 'VAE2':
        # the optimizer for the model
        optimizer = tf.keras.optimizers.Adam(1e-3)

        # model
        model = VAE2(
            enc=encoder,
            dec=decoder,
            optimizer=optimizer
        )

        # a pandas dataframe to save the loss information to
        losses = pd.DataFrame(columns=['recon_loss', 'latent_loss'])

    ##################################################################################
    print(f'##### Training')
    # for spec, index, indv in iter(dataset):
    #     model.train_net(x=spec)

    train_dataset = dataset
    # Todo
    #    create a real test set
    test_dataset = dataset

    # exampled data for plotting results
    example_data, _, _ = next(iter(train_dataset))
    example_data_reshaped = tf.cast(tf.reshape(example_data, (-1, *DIMS)), "float32") / 255.
    model.train_net(example_data_reshaped)

    n_epochs = 50
    for epoch in range(n_epochs):
        # train
        for batch, train_x in tqdm(
                zip(range(N_TRAIN_BATCHES), train_dataset), total=N_TRAIN_BATCHES
        ):
            data_train, ind, indv = train_x
            data_train_proc = tf.cast(tf.reshape(data_train, (-1, *DIMS)), "float32") / 255.
            model.train_net(data_train_proc)
        # test on holdout
        loss = []
        for batch, test_x in tqdm(
                zip(range(N_TEST_BATCHES), test_dataset), total=N_TEST_BATCHES
        ):
            data_test, ind, indv = test_x
            data_test_proc = tf.cast(tf.reshape(data_test, (-1, *DIMS)), "float32") / 255.
            loss.append(model.compute_loss(data_test_proc))
        losses.loc[len(losses)] = np.mean(loss, axis=0)
        # plot results
        print(
            "Epoch: {}".format(epoch)
        )
        if plot:
            if MODEL_TYPE == 'GAIA':
                model.plot_reconstruction(example_data)
            elif MODEL_TYPE == 'VAE':
                model.plot_reconstruction(example_data, N_Z)


if __name__ == '__main__':
    main()
