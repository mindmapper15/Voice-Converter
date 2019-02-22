# -*- coding: utf-8 -*-
# /usr/bin/python2

import glob
import random

import librosa
import numpy as np
import pyworld as pw
from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.dataflow.common import BatchData
from tensorpack.dataflow import PrefetchData
from audio import read_wav, preemphasis, amp2db, normalize_db
from hparam import hparam as hp

class DataFlowForConvert(RNGDataFlow):

    def __init__(self, data_path):
        self.wav_file = data_path
        self.batch_size = 1

    def __call__(self, n_prefetch=1, n_thread=1):
        df = self
        df = BatchData(df, 1)
        df = PrefetchData(df, n_prefetch, n_thread)
        return df

    def get_data(self):
        while True:
            yield get_mfccs_and_spectrogram(self.wav_file, isConverting=True, trim=False)


class DataFlow(RNGDataFlow):

    def __init__(self, data_path, batch_size):
        self.batch_size = batch_size
        self.wav_files = glob.glob(data_path)

    def __call__(self, n_prefetch=1000, n_thread=1):
        df = self
        df = BatchData(df, self.batch_size)
        df = PrefetchData(df, n_prefetch, n_thread)
        return df


class Net1DataFlow(DataFlow):

    def __init__(self, data_path, batch_size):
        self.batch_size = batch_size
        npz_path = data_path + '/*.npz'
        self.npz_files = glob.glob(npz_path)

    def get_data(self):
        while True:
            npz_file = random.choice(self.npz_files)
            #print(npz_file)
            yield read_mfccs_and_phones(npz_file)


class Net2DataFlow(DataFlow):

    def __init__(self, data_path, batch_size):
        self.batch_size = batch_size
        npz_path = data_path + '/npz/*.npz'
        self.npz_files = glob.glob(npz_path)

    def get_data(self):
        while True:
            npz_file = random.choice(self.npz_files)
            #print(npz_file)
            yield read_mfccs_and_spectral_envelope(npz_file)
            

def wav_random_crop(wav, sr, duration):
    assert (wav.ndim <= 2)

    target_len = sr * duration
    wav_len = wav.shape[-1]
    start = np.random.choice(range(np.maximum(1, wav_len - target_len)), 1)[0]
    end = start + target_len
    if wav.ndim == 1:
        wav = wav[start:end]
    else:
        wav = wav[:, start:end]
    return wav


def get_mfccs_and_phones(wav_file, trim=False, random_crop=True):

    '''This is applied in `train1` or `test1` phase.
    '''

    # Load
    wav = read_wav(wav_file, sr=hp.default.sr)

    mfccs = _get_mfcc(wav, hp.default.n_fft, hp.default.win_length, hp.default.hop_length)

    # timesteps
    num_timesteps = mfccs.shape[0]

    # phones (targets)
    phn_file = wav_file.replace("WAV", "PHN")
    phn2idx, idx2phn = load_vocab()
    phns = np.zeros(shape=(num_timesteps,))
    bnd_list = []
    for line in open(phn_file, 'r').read().splitlines():
        start_point, _, phn = line.split()
        bnd = int(start_point) // hp.default.hop_length
        phns[bnd:] = phn2idx[phn]
        bnd_list.append(bnd)

    # Trim
    if trim:
        start, end = bnd_list[1], bnd_list[-1]
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Random crop
    n_timesteps = (hp.default.duration * hp.default.sr) // hp.default.hop_length + 1
    if random_crop:
        start = np.random.choice(range(np.maximum(1, len(mfccs) - n_timesteps)), 1)[0]
        end = start + n_timesteps
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Padding or crop
    mfccs = librosa.util.fix_length(mfccs, n_timesteps, axis=0)
    phns = librosa.util.fix_length(phns, n_timesteps, axis=0)

    return mfccs, phns


def get_mfcc_and_spectral_envelope(wav_file, trim=True, random_crop=False, isConverting=False):
    '''This is applied in `train2`, `test2` or `convert` phase.
    '''
    # Load
    wav, _ = librosa.load(wav_file, sr=hp.default.sr, dtype=np.float64)

    # Trim
    if trim:
        wav, _ = librosa.effects.trim(wav, frame_length=hp.default.win_length, hop_length=hp.default.hop_length)

    if random_crop:
        wav = wav_random_crop(wav, hp.default.sr, hp.default.duration)


    # Padding or crop if not Converting
    if isConverting is False:
        length = int(hp.default.sr * hp.default.duration)
        wav = librosa.util.fix_length(wav, length)

    return _get_mfcc(wav, hp.default.n_fft, hp.default.win_length, hp.default.hop_length), _get_spectral_envelope(wav, hp.default.n_fft)


# TODO refactoring
def _get_mfcc(wav, n_fft, win_length, hop_length):

    # Pre-emphasis
    wav = preemphasis(wav, coeff=hp.default.preemphasis)

    # Get spectrogram
    D = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(D)

    # Get mel-spectrogram
    mel_basis = librosa.filters.mel(hp.default.sr, hp.default.n_fft, hp.default.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t) # mel spectrogram

    # Get mfccs, amp to db
    mel_db = amp2db(mel)
    mfccs = np.dot(librosa.filters.dct(hp.default.n_mfcc, mel_db.shape[0]), mel_db)

    return mfccs.T

def _get_spectral_envelope(wav, n_fft, f0_method='dio'):

    # Pre-emphasis
    wav = preemphasis(wav, coeff=hp.default.preemphasis)

    # Extract F0 info. Default Extraction method is dio-stonemask for speed.
    if f0_method == 'dio':
        f0, t_table = pw.dio(wav, hp.default.sr)
        f0 = pw.stonemask(wav, f0, t_table, hp.default.sr)
    elif f0_method == 'harvest':
        f0, t_table = pw.harvest(wav, hp.default.sr)

    # Extract Spectral Envelope
    spectral_envelope = pw.cheaptrick(wav, f0, t_table, hp.default.sr, fft_size = n_fft)

    # amp to db
    sp_en_db = librosa.amplitude_to_db(spectral_envelope)

    # Normalize Spectral Envelope to 0 ~ 1
    sp_en_db = normalize_db(sp_en_db, hp.default.max_db, hp.default.min_db)

    return sp_en_db.astype(np.float32)


def read_mfccs_and_phones(npz_file):
    np_arrays = np.load(npz_file)

    mfccs = np_arrays['mfccs']
    phns = np_arrays['phns']

    np_arrays.close()

    return mfccs, phns

def read_mfccs_and_spectral_envelope(npz_file):
    np_arrays = np.load(npz_file)

    mfccs = np_arrays['mfccs']
    sp_en = np_arrays['sp_en']

    np_arrays.close()

    return mfccs, sp_en



phns = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
        'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
        'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
        'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
        'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']


def load_vocab():
    phn2idx = {phn: idx for idx, phn in enumerate(phns)}
    idx2phn = {idx: phn for idx, phn in enumerate(phns)}

    return phn2idx, idx2phn
