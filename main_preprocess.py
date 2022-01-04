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
import click


@click.command()
@click.option("-n", "--dataset_id", type=str, help="Name of the database to chunk")
def main(dataset_id):
    if dataset_id == "bird-db":
        song_db = openBirdDB_df()
    else:
        song_db = None
    DT_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_path = Path(f"{DATA_DIR}/raw/{dataset_id}")
    wavs = set((dataset_path).expanduser().glob("**/*.wav"))
    print(f"Num files {len(wavs)}")
    for wf in wavs:
        if song_db is not None:
            generate_json(wf, DT_ID, song_db)
        else:
            generate_json_custom(wf, DT_ID)


if __name__ == "__main__":
    main()
    print("done")
