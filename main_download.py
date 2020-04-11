from sklearn.externals.joblib import Parallel, delayed
from tqdm import tqdm

import avgn
from avgn.downloading.birdDB import openBirdDB_df, downloadBirdDB
from avgn.utils.paths import DATA_DIR

song_db = openBirdDB_df()
bird_db_loc = f'{DATA_DIR}/raw/bird-db'
avgn.utils.paths.ensure_dir(bird_db_loc)

with Parallel(n_jobs=1, verbose=10) as parallel:
    parallel(delayed(downloadBirdDB)(row, bird_db_loc)
        for idx, row in tqdm(song_db.iterrows(), total=len(song_db)))


