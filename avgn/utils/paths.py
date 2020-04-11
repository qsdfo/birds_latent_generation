from pathlib2 import Path
import pathlib2
import os
from datetime import datetime
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
FIGURE_DIR = PROJECT_DIR / "figures"


def ensure_dir(file_path):
    """ create a safely nested folder
    """
    if type(file_path) == str:
        if "." in os.path.basename(os.path.normpath(file_path)):
            directory = os.path.dirname(file_path)
        else:
            directory = os.path.normpath(file_path)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except FileExistsError as e:
                # multiprocessing can cause directory creation problems
                print(e)
    elif type(file_path) == pathlib2.PosixPath:
        # if this is a file
        if len(file_path.suffix) > 0:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            file_path.mkdir(parents=True, exist_ok=True)


def most_recent_subdirectory(dataset_loc):
    """ return the subdirectory that has been generated most 
    recently with the "%Y-%m-%d_%H-%M-%S" time scheme used in AVGN
    """
    subdir_list = list((dataset_loc).iterdir())
    directory_dates = [
        datetime.strptime(i.name, "%Y-%m-%d_%H-%M-%S") for i in subdir_list
    ]
    return subdir_list[np.argsort(directory_dates)[-1]]
