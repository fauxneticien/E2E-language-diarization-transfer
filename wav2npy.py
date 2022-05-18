import os
import math
import python_speech_features as psf
import scipy.io.wavfile as wav
import numpy as np
import glob

# Liu et al: https://www.isca-speech.org/archive/pdfs/interspeech_2021/liu21d_interspeech.pdf

def extract_fbank(signal, sr, dim=23, log=True):
    high_frequency = sr / 2 - 200
    if log:
        features = psf.logfbank(signal, sr, nfilt=dim, lowfreq=20, highfreq=high_frequency, winlen=0.025, winstep=0.01)
    else:
        features = psf.fbank(signal, sr, nfilt=dim, lowfreq=20, highfreq=high_frequency)
    return features

wav_files = glob.glob("processed/*.wav")
# wav_files = ["4oLp3bc9OSJbDrwM_0000.wav"]
os.mkdir("npys")

for wav_file in wav_files:

    (wav_sr, wav_signal) = wav.read(wav_file)

    # clip duration in seconds
    wav_dur = len(wav_signal) / wav_sr

    # group signal into 200 ms to extract features, as per:
    # Liu et al: "Since the ground-truth language labels of the data in WSTCSMC 2020 are assigned to 200 ms segments, the input speech sample is first partitioned into segments of the same duration."

    # number of partitions (divide duration by 200 ms = 0.2 seconds)
    n_parts  = wav_dur / 0.2

    # Liu et al: "We ignore the remaining of the speech samples that cannot be divided exactly into 200ms segments."
    n_parts  = math.floor(n_parts)

    # Liu et al: "For each segment 23-dimensional log-Mel-filterbank features are extracted with 25 ms window and a 10 ms shift as [10] for all systems."

    # 10 ms shift over 200 ms  = 19 frames
    # 19 frames x 23-dim feats = 437 total features
    #
    # create placeholder of shape (p, 437), where p = number of partitions
    features = np.zeros((n_parts, 437))

    for ith_part in range(n_parts):

        start_sample = int(ith_part * 0.2 * wav_sr)
        end_sample   = int((start_sample + (0.2 * wav_sr) - 1))

        part_signal   = wav_signal[start_sample:end_sample]
        part_features = extract_fbank(part_signal, wav_sr)

        # Crop-frame in case overflow
        cropped_frame = part_features.shape[0] % 19

        if cropped_frame != 0:
            part_features = part_features[:-cropped_frame, :]

        features[ith_part,] = part_features.reshape(-1)

    # Save feature file
    npy_name = os.path.join("npys", os.path.basename(wav_file).rsplit('.')[0] + '.npy')
    np.save(npy_name, features)
