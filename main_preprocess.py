from datetime import datetime

from pathlib2 import Path

from avgn.custom_parsing.bird_db import generate_json
from avgn.downloading.birdDB import openBirdDB_df
from avgn.utils.paths import DATA_DIR


def main():
    song_db = openBirdDB_df()
    DATASET_ID = 'bird-db'
    DT_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    dataset_path = Path(f'{DATA_DIR}/raw/{DATASET_ID}')
    wavs = set((dataset_path).expanduser().glob('**/*.wav'))

    for wf in wavs:
        try:
            generate_json(wf, DT_ID, song_db)
        except:
            continue


if __name__ == '__main__':
    main()
