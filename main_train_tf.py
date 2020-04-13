import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import avgn.tensorflow.data as tfdata
from avgn.tensorflow.GAIA6 import GAIA
from avgn.utils.paths import DATA_DIR, ensure_dir


@click.command()
@click.option('-p', '--plot', is_flag=True)
def main(plot):
    ##################################################################################
    # Parameters
    DATASET_ID = 'Test_segmented'
    N_Z = 128
    TRAIN_BUF = 60000
    BATCH_SIZE = 64
    TEST_BUF = 10000
    DIMS = (28, 28, 1)
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
            image_dims = row.spectrogram.shape
            example = tfdata.serialize_example(
                {
                    "spectrogram": {
                        "data": row.spectrogram.flatten().tobytes(),
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
            ax[i].matshow(spec[i].numpy().reshape(image_dims), cmap=plt.cm.Greys, origin="lower")
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

    decoder = [
        tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu"),
        tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
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

    # the unet function
    gen_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.RMSprop(1e-3)

    # model
    model = GAIA(
        enc=encoder,
        dec=decoder,
        unet_function=unet_mnist,
        gen_optimizer=gen_optimizer,
        disc_optimizer=disc_optimizer,
        chsq=Chi2(df=1 / BATCH_SIZE)
    )

    ##################################################################################
    print(f'##### Training')
    for spec, index, indv in iter(dataset):
        model.train_net(x=spec)


def unet_convblock_up(
        last_conv,
        cross_conv,
        channels=16,
        kernel=(3, 3),
        activation="relu",
        pool_size=(2, 2),
        kernel_initializer="he_normal",
):
    """ A downsampling convolutional block for a UNET
    """

    up_conv = tf.keras.layers.UpSampling2D(size=(2, 2))(last_conv)
    merge = tf.keras.layers.concatenate([up_conv, cross_conv], axis=3)
    conv = tf.keras.layers.Conv2D(
        channels,
        kernel,
        activation=activation,
        padding="same",
        kernel_initializer=kernel_initializer,
    )(merge)
    conv = tf.keras.layers.Conv2D(
        channels,
        kernel,
        activation=activation,
        padding="same",
        kernel_initializer=kernel_initializer,
    )(conv)
    return conv


def unet_convblock_down(
        _input,
        channels=16,
        kernel=(3, 3),
        activation="relu",
        pool_size=(2, 2),
        kernel_initializer="he_normal",
):
    """ An upsampling convolutional block for a UNET
    """
    conv = tf.keras.layers.Conv2D(
        channels,
        kernel,
        activation=activation,
        padding="same",
        kernel_initializer=kernel_initializer,
    )(_input)
    conv = tf.keras.layers.Conv2D(
        channels,
        kernel,
        activation=activation,
        padding="same",
        kernel_initializer=kernel_initializer,
    )(conv)
    pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(conv)
    return conv, pool


def unet_mnist():
    """ the architecture for a UNET specific to MNIST
    """
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    up_1, pool_1 = unet_convblock_down(inputs, channels=32)
    up_2, pool_2 = unet_convblock_down(pool_1, channels=64)
    conv_middle = tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(pool_2)
    conv_middle = tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(conv_middle)
    down_2 = unet_convblock_up(conv_middle, up_2, channels=64)
    down_1 = unet_convblock_up(down_2, up_1, channels=32)
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(down_1)
    return inputs, outputs


if __name__ == '__main__':
    main()
