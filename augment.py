# This file augments data by resynthesizing data across different files and clips.
# Assumes wavdata/ folder and train.txt is already present.

import glob
import os

import numpy as np
import pandas as pd
import random
import soundfile as sf

from math import floor
from pympi import Praat
from statistics import multimode

def get_annotated_intervals(tg_data, tier_name):
    # Praat TextGrids contains both empty and annotated intervals, e.g.
    # [(0, 1, ''), (2, 3, 'not empty')], so discard ones with empty text

    # Also convert to times to milliseconds for convenience:
    # [(2, 3, 'not empty')] => [(2000, 3000, 'not empty')]

    return [ (int(start_sec * 1000), int(end_sec * 1000), text) for start_sec, end_sec, text in tg_data.get_tier(tier_name).get_intervals() if len(text.strip()) > 0 ]

training_clips = set()
with open('train.txt' , 'r') as f:
    for l in f.readlines():
        # binary_npys/VU6..._0012.npy 101101011... 20
        path = l.split()[0].rsplit('.')[0].rsplit('/')[-1]
        training_clips.add(path)

#textgrid_files = ['wavdata/4xyIm2P6Xzlin341.TextGrid', 'wavdata/ZQC2TFqLsvNDq9bP.TextGrid']
textgrid_files = glob.glob("wavdata/*.TextGrid")
wav_and_label = []
# Synthesis assumes sr is the same across the dataset.
sr = None
for tg_file in textgrid_files:
    tg_basename = os.path.basename(tg_file).rsplit('.')[0]
    wav_file = tg_file.rsplit('.')[0] + ".wav"
    wav_data, wav_sr = sf.read(wav_file)
    if sr == None:
        sr = wav_sr

    tg_data = Praat.TextGrid(file_path=tg_file)

    clips   = get_annotated_intervals(tg_data, "clip")
    speech  = get_annotated_intervals(tg_data, "speech")
    english = get_annotated_intervals(tg_data, "english")

    for clip_index, clip_info in enumerate(clips):
        # skip other splits
        if tg_basename + '_' + str(clip_index) .zfill(4) not in training_clips:
            continue
        
        clip_start, clip_end, _ = clip_info

        # subset the wav file to get samples for clip
        clip_wav = []
        # Get annotations from 'speech' tier that occur within the clip, and normalise times relative to the start of the clip
        speech_intervals = [ (max(clip_start, start_ms) - clip_start, min(clip_end, end_ms) - clip_start) for start_ms, end_ms, _ in speech if end_ms >= clip_start and start_ms <= clip_end ]

        # Initialize clip_ones with all zeros
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        clip_zeros = np.zeros(clip_end - clip_start)

        # Set times with any speech to 1
        # [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
        for start_ms, end_ms in speech_intervals:
            clip_zeros[start_ms:end_ms] = 1
 
        clip_dur_ms = clip_zeros.sum()

        # Skip silent clips
        if clip_dur_ms == 0:
            continue

        # Get annotations from 'english' tier that occur within the clip, and normalise times relative to the start of the clip
        english_intervals = [ (max(clip_start, start_ms) - clip_start, min(clip_end, end_ms) - clip_start) for start_ms, end_ms, _ in english if end_ms >= clip_start and start_ms <= clip_end ]

        # Set times with English speech to 2
        # [1, 1, 1, 1, 2, 2, 1, 1, 1, 1]
        clip_twos = np.zeros(clip_end - clip_start)
        for start_ms, end_ms in english_intervals:
            clip_twos[start_ms:end_ms] = 1
        clip_zeros += clip_zeros * clip_twos

        # Slices the clip into continuous 0 or 1s.
        start_ptr = 0
        end_ptr = 1
        while True:
            while start_ptr < len(clip_zeros) and clip_zeros[start_ptr] == 0:
                start_ptr += 1
            if start_ptr >= len(clip_zeros):
                break
            end_ptr = start_ptr + 1
            while end_ptr <= len(clip_zeros) and clip_zeros[start_ptr] == clip_zeros[end_ptr-1]:
                end_ptr += 1
            wav_frac = wav_data[int((start_ptr+clip_start) / 1000 * wav_sr) : int((end_ptr+clip_start) / 1000 * wav_sr)]
            # map 2 (eng) to label 1 and 1 (Hindi) to 0
            wav_label = np.zeros(end_ptr - start_ptr, int)
            if clip_zeros[start_ptr] == 2:
                wav_label = np.ones(end_ptr - start_ptr, int)
            wav_and_label.append((wav_frac, wav_label))
            start_ptr = end_ptr

random.shuffle(wav_and_label)

clip_wavs = []
clip_labels = []
srs = []
start_ptr = 0
while start_ptr < len(wav_and_label):
    # Synthesizes 0 language and 1 language clips together with random length.
    dice = random.randint(5, 15)
    end_ptr = start_ptr + dice if start_ptr + dice < len(wav_and_label) else len(wav_and_label)
    syn_wav = [data for section in wav_and_label[start_ptr:end_ptr] for data in section[0]]
    # Truncates into multiples of 200ms.
    if len(syn_wav) % 200 != 0:
        syn_wav = syn_wav[0:-(len(syn_wav) % 200)]
    clip_wavs.append(syn_wav)

    syn_label = [data for section in wav_and_label[start_ptr:end_ptr] for data in section[1]]
    if len(syn_label) % 200 != 0:
        syn_label = syn_label[0:-(len(syn_label) % 200)]
    syn_label = np.array_split(syn_label, len(syn_label) / 200)
    syn_label = [ str(multimode(group)[0]) for group in syn_label ]
    syn_label = "".join(syn_label)
    clip_labels.append(syn_label)
    start_ptr += dice

if not os.path.isdir('augmented_data/'):
    os.mkdir('augmented_data')

textlines = []
for i in range(len(clip_wavs)):
    # write out wav clip
    id_str = "aug_" + str(i)
    sf.write("augmented_data/" + id_str + ".wav", clip_wavs[i], sr)
    textlines.append(os.path.join("augmented_npys", f'{id_str+".npy"} {clip_labels[i]} {len(clip_labels[i])}'))

with open('aug_train.txt', 'w') as f:
    f.write('\n'.join(textlines))

