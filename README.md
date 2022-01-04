# Birds latent generation

# Installations

    pip -r requirements.txt

# Create a database

## Birds_DB
To use with Bird_db, you can download the files by running

    python main_download.py

If BIRD_DB.xls is missing it's here (https://github.com/timsainb/AVGN/blob/master/notebooks/birdsong/cassins_vireo_example/BIRD_DB.xls)

## Custom database
If you plan to use a custom database, perhaps it is a good idea to first get used to the structure of the database by downloading and running the code on birds_db (see previous section).

The database should have the following structure, (with the exact names, unless indicated):
```
birds_latent_generation
│   README.md
│   avgn
│   ...
└───data
    │   BIRD_DB.xls (only if you work with BIRD_DB)
    └───raw
        └───database_name (or custom name)
            └───label_1 (or custom name)
            │   └───wavs
            │   │   │   "name1 (or custom name).wav"
            │   │   │   "name2.wav"
            │   │   │   ...
            │   │
            │   └───TextGrids (optional)
            │       │   "name1 (or custom name, but same as in the corresponding wavs folder).TextGrid"
            │       │   "name2.TextGrid"
            │       │   ...
            │
            └───label_2 (or custom name)
            │   ...

```

TextGrid information is optional and correspond to manual annotations of segments.
If not provided, automatic segmentation will be performed later.
The name for wav and corresponding textgrid files have to be exactly the same.

Each subfolder label_1, label_2 represents class of data.

Here is an example of a TextGrid file (only the first 2 segments are shown):

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


# Processing the database
## Chunk files (optional)
Depending on your computer's RAM and recordings length, spectrograms may not fit in memory.
In that case run, you can chunk files in a database by running

    python main_chunk.py -n database_name -d 120

Argument -n is the name of the database.
Argument -d is is the duration for chunks in seconds (you can omit this argument which will default to 120 seconds)

It will create another database folder with "_chunks" appended at the end of the name.
The code does not deal with TextGrids for now, so you will need to perform automatic segmentation as described below.

## Preprocess
Create a json file listing the wav files in the database

    python main_preprocess.py -n database_name

This will create another processed folder where it stores JSON files containing information about both the database and the processing.

    birds_latent_generation
    └───data
        └───raw
        └───processed
            └───name
                └───id (based on date and time)
                    └───...

## Segment (optional)
This step is useless if your database has been manually annotated (this is the case for bird_db).
Otherwise, it performs automatic segmentation.

    python main_segment.py -n database_name

- main_spectrogram
- train and generate with main_torch