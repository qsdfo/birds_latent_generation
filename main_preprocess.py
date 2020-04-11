from datetime import datetime
import click
import glob
import numpy as np
from pathlib2 import Path
from tqdm import tqdm

from avgn.custom_parsing.bird_db import generate_json

from avgn.utils.paths import DATA_DIR, ensure_dir
from avgn.downloading.birdDB import openBirdDB_df
from sklearn.externals.joblib import Parallel, delayed


@click.command()
@click.option('-n', '--n_jobs', type=int, default=1)
def main(n_jobs):
    song_db = openBirdDB_df()
    DATASET_ID = 'bird-db'
    DT_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    dataset_path = Path(f'{DATA_DIR}/raw/{DATASET_ID}')
    wavs = set((dataset_path).expanduser().glob('**/*.wav'))

    for wf in tqdm(wavs):
        try:
            generate_json(wf, DT_ID, song_db)
        except:
            continue
    # with Parallel(n_jobs=n_jobs, verbose=10) as parallel:
    #     parallel(
    #         delayed(
    #
    #     )


if __name__ == '__main__':
    main()