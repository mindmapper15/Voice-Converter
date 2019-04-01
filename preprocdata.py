import os
import glob
import librosa
import numpy as np
import datetime
from numpy import savez
from hparam import hparam as hp
import pyworld as pw
from data_load import get_mfcc_and_spectral_envelope

def generate_npz(wav_files):
    for i in range(len(wav_files)):
        f_name = wav_files[i].replace('wav', 'npz')
        mfccs, spectral_envelope = get_mfcc_and_spectral_envelope(wav_files[i])
        savez(f_name, mfccs=mfccs, sp_en=spectral_envelope)


def matching_list(wav_files, npz_files):
    not_converted = []
    for i in range(len(wav_files)):
        npz_name = wav_files[i].replace('wav','npz')
        if npz_name not in npz_files:
            not_converted.append(wav_files[i])

    for i in range(len(npz_files)):
        wav_name = npz_files[i].replace('npz', 'wav')
        if wav_name not in wav_files:
            os.remove(npz_files[i])

    return not_converted


def extract_f0(logdir_path, wav_files, method='dio', num_speciment=250):
    f0_set = list()
    fs = hp.default.sr

    if num_speciment > len(wav_files):
        num_speciment = len(wav_files)

    for i in range(num_speciment):
        y, _ = librosa.load(wav_files[i], sr=fs, dtype=np.float64)
        y, _ = librosa.effects.trim(y, frame_length=hp.default.win_length, hop_length=hp.default.hop_length)

        # Extract F0 Information from each WAV files
        if method == 'dio':
            _f0, t = pw.dio(y, fs)
            f0 = pw.stonemask(y, _f0, t, fs)
        elif method == 'harvest':
            f0, t = pw.harvest(y, fs)
        else:
            raise ParameterError('Invalid method specification: {}'.format(method))

        f0 = f0[f0!=0]
        f0_set.extend(f0)

    f0_set = np.array(f0_set)

    f0_mean = np.mean(f0_set)
    f0_variance = np.var(f0_set)
    f0_std = np.std(f0_set)

    if len(glob.glob(logdir_path)) == 0:
        os.makedirs(logdir_path)

    np.savez(logdir_path + '/f0_info.npz', f0_mean=f0_mean, f0_variance=f0_variance, f0_std=f0_std)


def preprocessing(dataset_path, logdir_path, isConverting=False):

    if isConverting is False:
        dataset_path_origin = dataset_path
        dataset_path = dataset_path + '/wav/*.wav'
        if os.path.isdir(dataset_path_origin + '/npz') == False:
            os.makedirs(dataset_path_origin + '/npz')

    s = datetime.datetime.now()

    wav_files = glob.glob(dataset_path)
    dataset_path = dataset_path.replace('wav', 'npz')
    npz_files = glob.glob(dataset_path)

    if len(npz_files) is 0:
        generate_npz(wav_files)
    else:
        convert_list = matching_list(wav_files, npz_files)

        if len(convert_list) is 0:
            print('All WAV files in dataset directory are already converted!')
        else:
            generate_npz(convert_list)

    if len(glob.glob(logdir_path + '/f0_info.npz')) == 0:
        print('There is no f0 information in dataset directory')
        print('Extracting f0 information from dataset...')
        extract_f0(logdir_path, wav_files)
    else:
        print('F0 informations are already extracted!')

    e = datetime.datetime.now()
    diff = e - s
    print("Done. elapsed time:{}s".format(diff.seconds))
