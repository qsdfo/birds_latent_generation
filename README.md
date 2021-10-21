# Birds latent generation

## Installation

    pip -r requirements.txt

## Create a database

### Birds_DB
To use with Bird_db, you can download the files by running

    python main_download.py

If BIRD_DB.xls is missing it's here (https://github.com/timsainb/AVGN/blob/master/notebooks/birdsong/cassins_vireo_example/BIRD_DB.xls)

### Custom database
The database should have the following structure:
birds_latent_generation > data > raw > "name of database" > wavs > "audio files with .wav extension"

If the dataset has been manually annotated, you can provide segmentation information in textgrids files
birds_latent_generation > data > raw > "name of database" > TextGrids > "same name as corresponding wav but with .TextGrid extension"

An exemple of TextGris file for 2 segments is:

        File type = "ooTextFile"
        Object class = "TextGrid"

        xmin = 0
        xmax = 988.0566666666667
        tiers? <exists>
        size = 1
        item []:
            item [1]:
                class = "IntervalTier"
                name = "Cavi1113"
                xmin = 0
                xmax = 988.0566666666667
                intervals: size = 307
                intervals [1]:
                    xmin = 0
                    xmax = 68.67543075919981
                    text = ""
                intervals [2]:
                    xmin = 68.67543075919981
                    xmax = 68.9732043596961
                    text = "ah"


## Chunk files
Recordings can be too long for there spectrograms to fit in memory. Depending on your computer's RAM, you may need to chunk files first.
In that case run

    python utils/chunk_files

Duration is hard-coded in the function... but it's quite easy to find it (chunk_duration_second)

## Calibrate decibel
You may want to run main_calibrate_db to find the ideal values for the ref and min dB parameters

    python main_calibrate_db.py

## Preprocess
Create a json file listing the files

    python main_preprocess.py

## Segment
This step is useless if your database has been manually annotated (this is the case for BIRD_DB).
Otherwise, it performs automatic segmentation and populate the

    python main_segment.py

- main_spectrogram
- train and generate with main_torch