"""
Create for each wav file a JSON containing meta information.
If segmentation has been manually done, it also contains start and end of syllables.
If not, it is necessary to run main_segment.py
"""
from datetime import datetime

from pathlib2 import Path

from avgn.custom_parsing.bird_db import generate_json, generate_json_custom
from avgn.downloading.birdDB import openBirdDB_df
from avgn.utils.paths import DATA_DIR


def main():
    # song_db = openBirdDB_df()
    # DATASET_ID = 'bird-db'
    # DT_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    song_db = None
    DATASET_ID = 'voizo_chunks'
    DT_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    dataset_path = Path(f'{DATA_DIR}/raw/{DATASET_ID}')
    wavs = set((dataset_path).expanduser().glob('**/*.wav'))

    for wf in wavs:
        if song_db is not None:
            generate_json(wf, DT_ID, song_db)
        else:
            generate_json_custom(wf, DT_ID)

if __name__ == '__main__':
    main()
    print('done')
