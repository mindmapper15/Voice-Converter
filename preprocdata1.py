import os
import glob
import librosa
import numpy as np
import datetime
from numpy import savez
from hparam import hparam as hp
import pyworld as pw
from data_load import get_mfccs_and_spectrogram, get_mfccs_and_phones

def generate_npz(wav_files):
    n_total=len(wav_files)
    for i in range(n_total):
        if (i % (n_total//10)) == 0:
            print((i / (n_total//10))*10,"% is converted")
            
        mfccs, phns = get_mfccs_and_phones(wav_files[i])
        f_name = wav_files[i].replace('WAV', 'npz')
        savez(f_name, mfccs=mfccs, phns=phns)
        

def matching_list(wav_files, npz_files):
    not_converted = []
    for i in range(len(wav_files)):
        npz_name = wav_files[i].replace('WAV','npz')
        if npz_name not in npz_files:
            not_converted.append(wav_files[i])
            
    for i in range(len(npz_files)):
        wav_name = npz_files[i].replace('npz', 'WAV')
        if wav_name not in wav_files:
            os.remove(npz_files[i])

    return not_converted


def preprocessing(dataset_path, isConverting=False):

    
    
    s = datetime.datetime.now()
    
    wav_files = glob.glob(dataset_path)
    dataset_path = dataset_path.replace('WAV', 'npz')
    npz_files = glob.glob(dataset_path)

    if len(npz_files) is 0:
        generate_npz(wav_files)
    else:
        convert_list = matching_list(wav_files, npz_files)

        if len(convert_list) is 0:
            print('All WAV files in dataset directory are already converted!')
        else:
            generate_npz(convert_list)
    
    e = datetime.datetime.now()
    diff = e - s
    print("Done. elapsed time:{}s".format(diff.seconds))
