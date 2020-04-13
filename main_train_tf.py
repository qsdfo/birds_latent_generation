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
    print(f'##### Build training dataset')
    DATASET_ID = 'Test_segmented'
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
    model = GAIA()

    ##################################################################################
    print(f'##### Training')
    for spec, index, indv in iter(dataset):
        model.train_net(x=spec)


if __name__ == '__main__':
    main()