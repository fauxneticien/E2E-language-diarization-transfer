import glob
import os

import soundfile as sf
import numpy as np
import pandas as pd

from math import floor
from pympi import Praat
from statistics import multimode

def get_annotated_intervals(tg_data, tier_name):
    # Praat TextGrids contains both empty and annotated intervals, e.g.
    # [(0, 1, ''), (2, 3, 'not empty')], so discard ones with empty text

    # Also convert to times to milliseconds for convenience:
    # [(2, 3, 'not empty')] => [(2000, 3000, 'not empty')]

    return [ (int(start_sec * 1000), int(end_sec * 1000), text) for start_sec, end_sec, text in tg_data.get_tier(tier_name).get_intervals() if len(text.strip()) > 0 ]

# Create placeholder for dataframes
tg_dfs = []

for tg_file in ["4oLp3bc9OSJbDrwM.TextGrid", "4xyIm2P6Xzlin341.TextGrid"]:

    # 4oLp3bc9OSJbDrwM.TextGrid => 4oLp3bc9OSJbDrwM
    tg_basename = os.path.basename(tg_file).rsplit('.')[0]
    
    # wav_file = tg_file.rsplit(".")[0] + ".wav"
    # wav_data, wav_sr = sf.read(wav_file)

    tg_data = Praat.TextGrid(file_path=tg_file)

    clips   = get_annotated_intervals(tg_data, "clip")
    speech  = get_annotated_intervals(tg_data, "speech")
    english = get_annotated_intervals(tg_data, "english")

    clip_labels = []

    for clip_index, clip_info in enumerate(clips):
        
        clip_start, clip_end, _ = clip_info

        # 4oLp3bc9OSJbDrwM => 4oLp3bc9OSJbDrwM_0001.wav
        clip_id = tg_basename + + "_" + str(clip_index).zfill(4)
        
        # subset the wav file to get samples for clip
        # start_sample = clip_start * wav_sr
        # end_sample = clip_end * wav_sr
        # clip_wav = wav_data[start_sample:end_sample]
        # sf.write(clip_id + '.wav', clip_wav, wav_sr)

        clip_dur_ms = clip_end - clip_start

        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        clip_zeros  = np.zeros(clip_dur_ms, dtype=np.int8)

        # Get annotations from 'speech' tier that occur within the clip, and normalise times relative to the start of the clip
        clip_ones = [ (start_ms - clip_start, end_ms - clip_start) for start_ms, end_ms, _ in speech if start_ms >= clip_start and end_ms <= clip_end ]

        # Set times with any speech to 1
        # [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
        for start_ms, end_ms in clip_ones:
            clip_zeros[start_ms:end_ms] = 1
            
        # Get annotations from 'english' tier that occur within the clip, and normalise times relative to the start of the clip
        clip_twos = [ (start_ms - clip_start, end_ms - clip_start) for start_ms, end_ms, _ in english if start_ms >= clip_start and end_ms <= clip_end ]

        # Set times with English speech to 2
        # [0, 0, 1, 1, 2, 2, 1, 1, 0, 0]
        for start_ms, end_ms in clip_twos:
            clip_zeros[start_ms:end_ms] = 2

        # Bin times into 200-millisecond chunks (and get most frequently occurring value for bin)
        # to create labels required for: https://github.com/Lhx94As/E2E-language-diarization/blob/main/data/data.txt

        n_bins = floor(clip_dur_ms / 200)

        label_arr = np.array_split(clip_zeros, n_bins)

        # FYI multimode requires python >= 3.8
        # Use multimode in case of ties and just pick the first label as mode
        label_arr = [ str(multimode(bin)[0]) for bin in label_arr ]
        label_str = "".join(label_arr)

        clip_labels.append(label_str)

    tg_dfs.append(pd.DataFrame({
        "clip_id" : [ tg_basename + "_" + str(i).zfill(4) for i in range(len(clips)) ],
        "text" : [ text for _, _, text in clips ],
        "label" : clip_labels
    }))

labels_df = pd.concat(tg_dfs)

print(labels_df[['clip_id', 'label']])

# labels_df.to_csv("labels.tsv", sep=" ", index=False)
