import argparse
import os
import math
import python_speech_features as psf
import scipy.io.wavfile as wav
import torchaudio
import numpy as np
import glob

from speechbrain.pretrained import EncoderClassifier

# Liu et al: https://www.isca-speech.org/archive/pdfs/interspeech_2021/liu21d_interspeech.pdf

def extract_fbank(signal, sr, dim=23, log=True):
    high_frequency = sr / 2 - 200
    if log:
        features = psf.logfbank(signal, sr, nfilt=dim, lowfreq=20, highfreq=high_frequency, winlen=0.025, winstep=0.01)
    else:
        features = psf.fbank(signal, sr, nfilt=dim, lowfreq=20, highfreq=high_frequency)
    return features


def get_wav_features(wav_file, feature, language_id):
    wav_sr = 16000
    wav_signal = None
    ft_dim = 0
    if feature == 'fbank':
        (wav_sr, wav_signal) = wav.read(wav_file)
        # Liu et al: "For each segment 23-dimensional log-Mel-filterbank features are extracted with 25 ms window and a 10 ms shift as [10] for all systems."

        # 10 ms shift over 200 ms  = 19 frames
        # 19 frames x 23-dim feats = 437 total features
        ft_dim = 437

    elif feature == 'etdnn':
        wav_signal = language_id.load_audio(wav_file)
        ft_dim = 256
    else:
        print('Unrecognized feature type: ', feature)
        return

    # clip duration in seconds
    wav_dur = len(wav_signal) / wav_sr

    # group signal into 200 ms to extract features, as per:
    # Liu et al: "Since the ground-truth language labels of the data in WSTCSMC 2020 are assigned to 200 ms segments, the input speech sample is first partitioned into segments of the same duration."

    # number of partitions (divide duration by 200 ms = 0.2 seconds)
    n_parts  = wav_dur / 0.2

    # Liu et al: "We ignore the remaining of the speech samples that cannot be divided exactly into 200ms segments."
    n_parts  = math.floor(n_parts)

    features = np.zeros((n_parts, ft_dim))

    for ith_part in range(n_parts):

        start_sample = int(ith_part * 0.2 * wav_sr)
        end_sample   = int(start_sample + (0.2 * wav_sr))

        part_signal   = wav_signal[start_sample:end_sample]
        part_features = None
        if feature == 'fbank':
            part_features = extract_fbank(part_signal, wav_sr)
        elif feature == 'etdnn':
            part_features = language_id.encode_batch(part_signal)

        # Crop-frame in case overflow
        #cropped_frame = part_features.shape[0] % 19

        #if cropped_frame != 0:
        #    part_features = part_features[:-cropped_frame, :]

        features[ith_part,] = part_features.reshape(-1)
    return features

def main():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--savedir', type=str, default='new_npys')
    parser.add_argument('--feature', type=str, help='etdnn or fbank', default='fbank')

    args = parser.parse_args()

    wav_files = glob.glob("binary_processed/*.wav")
    #wav_files = glob.glob("augmented_data/*.wav")
    #wav_files = ["binary_processed/4oLp3bc9OSJbDrwM_0022.wav"]

    # savedir can be e.g. binary_npys_ecapa_tdnn
    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)

    # Only used for ecapa-tdnn.
    language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")

    for wav_file in wav_files:
        features = get_wav_features(wav_file, args.feature, language_id)
        # Save feature file
        npy_name = os.path.join(args.savedir, os.path.basename(wav_file).rsplit('.')[0] + '.npy')
        np.save(npy_name, features)



if __name__ == "__main__":
    main()
