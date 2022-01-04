from tqdm import tqdm

import avgn
from avgn.downloading.birdDB import openBirdDB_df, downloadBirdDB
from avgn.utils.paths import DATA_DIR


def main():
    song_db = openBirdDB_df()
    bird_db_loc = f"{DATA_DIR}/raw/bird-db"
    avgn.utils.paths.ensure_dir(bird_db_loc)
    for _, row in tqdm(song_db.iterrows(), total=len(song_db)):
        try:
            downloadBirdDB(row, bird_db_loc)
        except:
            print(f"Corrupted {row.Audio_file}")


if __name__ == "__main__":
    main()
