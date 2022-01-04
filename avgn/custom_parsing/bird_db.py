import librosa
from avgn.utils.json_custom import NoIndent, NoIndentEncoder
import pandas as pd
from datetime import datetime
from praatio import tgio
from avgn.utils.paths import DATA_DIR, ensure_dir
from avgn.utils.audio import get_samplerate
import json
from datetime import time as dtt


def generate_json_custom(wavfile, DT_ID):
    indv = wavfile.parent.parent.stem
    dataset_id = wavfile.parent.parent.parent.stem
    wav_loc = wavfile.as_posix()
    dt = datetime.now()
    datestring = dt.strftime("%Y-%m-%d")

    DATASET_ID = f"{dataset_id}"
    sr = get_samplerate(wavfile.as_posix())
    wav_duration = librosa.get_duration(filename=wavfile.as_posix())
    wav_loc = wavfile.as_posix()

    # make json dictionary
    json_dict = {
        "sample_rate": sr,
        "species": indv,
        "datetime": datestring,
        "wav_loc": wav_loc,
        "samplerate_hz": sr,
        "length_s": wav_duration,
    }

    # no manual segmentation
    json_dict["indvs"] = {
        indv: {
            "syllables": {
                "start_times": [],
                "end_times": [],
                "labels": [],
            }
        }
    }

    # generate json
    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    json_out = (
        DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (wavfile.stem + ".JSON")
    )

    # save json
    ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))
    return


def generate_json(wavfile, DT_ID, song_db):
    indv = wavfile.parent.parent.stem
    try:
        dt = datetime.strptime(wavfile.stem, "%Y-%m-%d_%H-%M-%S-%f")
    except ValueError:
        dt = datetime.now()
    datestring = dt.strftime("%Y-%m-%d")

    row = song_db[
        (song_db.SubjectName == indv)
        & (song_db.recording_date == datestring)
        & (song_db.recording_time == dt.time())
    ].iloc[0]

    # make json dictionary
    json_dict = {}
    for key in dict(row).keys():
        if type(row[key]) == pd._libs.tslibs.timestamps.Timestamp:
            json_dict[key] = row[key].strftime("%Y-%m-%d_%H-%M-%S")
        elif type(row[key]) == dtt:
            json_dict[key] = row[key].strftime("%H:%M:%S")
        elif type(row[key]) == pd._libs.tslibs.nattype.NaTType:
            continue
        else:
            json_dict[key] = row[key]

    species_name = row.Species_short_name.replace(" ", "_")
    common_name = row.Subject_species.replace(" ", "_")
    DATASET_ID = "BIRD_DB_" + species_name

    json_dict["species"] = species_name
    json_dict["common_name"] = common_name
    json_dict["datetime"] = datestring

    sr = get_samplerate(wavfile.as_posix())
    wav_duration = librosa.get_duration(filename=wavfile.as_posix())

    json_dict["wav_loc"] = wavfile.as_posix()
    # rate and length
    json_dict["samplerate_hz"] = sr
    json_dict["length_s"] = wav_duration

    tg = wavfile.parent.parent / "TextGrids" / (wavfile.stem + ".TextGrid")

    if not tg.exists():
        print(tg.as_posix(), "File does not exist")
        return
    textgrid = tgio.openTextgrid(fnFullPath=tg)

    tierlist = textgrid.tierDict[textgrid.tierNameList[0]].entryList
    start_times = [i.start for i in tierlist]
    end_times = [i.end for i in tierlist]
    labels = [i.label for i in tierlist]

    json_dict["indvs"] = {
        indv: {
            "syllables": {
                "start_times": NoIndent(start_times),
                "end_times": NoIndent(end_times),
                "labels": NoIndent(labels),
            }
        }
    }

    # generate json
    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    json_out = (
        DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (wavfile.stem + ".JSON")
    )

    # save json
    ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))
