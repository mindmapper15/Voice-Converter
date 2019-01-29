# -*- coding: utf-8 -*-
# /usr/bin/python2


from __future__ import print_function

import argparse
import os
import glob
import matplotlib.pyplot as plt

from models import Net2ForConvert
import numpy as np
from audio import spec2wav_lws, inv_preemphasis, db2amp, denormalize_db, f0_adapt
import datetime
import tensorflow as tf
from librosa.output import write_wav
from librosa.core import load
from hparam import hparam as hp
from data_load import get_mfccs_and_spectrogram
from tensorflow.python.client import timeline
from tensorpack.predict.base import OfflinePredictor
from tensorpack.predict.config import PredictConfig
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.tfutils.sessinit import ChainInit
from tensorpack.callbacks.base import Callback


def convert(predictor, mfcc, spec, mel_spec):
    print("convert")
    pred_s = datetime.datetime.now()
    pred_spec, _, ppgs = predictor(mfcc, spec, mel_spec)
    pred_e = datetime.datetime.now()
    pred_t = pred_e - pred_s
    print("Predicting time:{}s".format(pred_t.seconds))

    preproc_s = datetime.datetime.now()
    # Denormalizatoin
    print("denormalize_db")
    pred_spec = denormalize_db(pred_spec, hp.default.max_db, hp.default.min_db)

    # Db to amp
    print("db2amp")
    pred_spec = db2amp(pred_spec)

    # Emphasize the magnitude
    print("emphasize")
    pred_spec = np.power(pred_spec, hp.convert.emphasis_magnitude)

    preproc_e = datetime.datetime.now()
    preproc_t = preproc_e - preproc_s
    print("Pre-Processing time:{}s".format(preproc_t.seconds))

    audio = []
    # Spectrogram to waveform
    recon_s = datetime.datetime.now()

    print("spec2wav")
    audio.append(spec2wav_lws(pred_spec[0], hp.default.n_fft, hp.default.win_length, hp.default.hop_length, hp.default.lws_mode))
    recon_e = datetime.datetime.now()
    recon_t = recon_e - recon_s
    print("Converting Spectrogram-to-Wave time:{}s".format(recon_t.seconds))


    audio = np.array(audio)
    # print('audio.shape : ', audio.shape)

    # Apply inverse pre-emphasis
    audio = inv_preemphasis(audio, coeff=hp.default.preemphasis)
    return audio[0], ppgs


def get_eval_input_names():
    return ['x_mfccs', 'y_spec', 'y_mel']


def get_eval_output_names():
    return ['pred_spec', 'y_spec', 'ppgs']


def do_convert(args, logdir1, logdir2):
    print("do_convert")
    # Load graph
    
    ckpt1 = tf.train.latest_checkpoint(logdir1)
    ckpt2 = '{}/{}'.format(logdir2, args.ckpt) if args.ckpt else tf.train.latest_checkpoint(logdir2)
    model = Net2ForConvert()

    session_inits = []
    if ckpt2:
        session_inits.append(SaverRestore(ckpt2))
    if ckpt1:
        session_inits.append(SaverRestore(ckpt1, ignore=['global_step']))
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    print("PredictConfig")
    pred_conf = PredictConfig(
        model=model,
        input_names=get_eval_input_names(),
        output_names=get_eval_output_names(),
        session_init=ChainInit(session_inits)
    )

    print("OfflinePredictor")
    set_env_s = datetime.datetime.now()
    predictor = OfflinePredictor(pred_conf)
    set_env_e = datetime.datetime.now()
    set_env_t = set_env_e - set_env_s
    print("Setting Environment time:{}s".format(set_env_t.seconds))

    input_name = ''
    while True:
        input_name = input("Write your audio file\'s path for converting : ")
        if input_name == 'quit':
            break
        elif len(glob.glob(input_name)) == 0:
            print("That audio file doesn't exist! Try something else.")
            continue

        convert_s = datetime.datetime.now()
        mfcc, spec, mel_spec = get_mfccs_and_spectrogram(input_name, trim=False, isConverting=True)
        mfcc = np.expand_dims(mfcc, axis=0)
        spec = np.expand_dims(spec, axis=0)
        mel_spec = np.expand_dims(mel_spec, axis=0)
        output_audio, ppgs = convert(predictor, mfcc, spec, mel_spec)

        input_audio, samplerate = load(input_name, sr=hp.default.sr, dtype=np.float64)

        """
        # F0 adaptation with WORLD Vocoder
        f0_conv_s = datetime.datetime.now()
        output_audio = f0_adapt(input_audio, output_audio, logdir2, samplerate)
        f0_conv_e = datetime.datetime.now()
        f0_conv_time = f0_conv_e - f0_conv_s
        print("F0 Adapting Time:{}s".format(f0_conv_time.seconds))
        """

        # Saving voice-converted audio to 32-bit float wav file
        # print(audio.dtype)
        output_audio = output_audio.astype(np.float32)
        write_wav(path="./converted/"+input_name,y=output_audio,sr=hp.default.sr)

        # Saving PPGS data to Grayscale Image and raw binary file
        ppgs = np.squeeze(ppgs, axis=0)
        plt.imsave('./converted/debug/'+input_name+'.png', ppgs, cmap='binary')
        np.save('./converted/debug/'+input_name+'.npy', ppgs)
        
        convert_e = datetime.datetime.now()
        convert_time = convert_e - convert_s
        print("Total Converting Time:{}s".format(convert_time.seconds))

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('case1', type=str, help='experiment case name of train1')
    parser.add_argument('case2', type=str, help='experiment case name of train2')
    parser.add_argument('-ckpt', help='checkpoint to load model.')
    parser.add_argument('-gpu', help='comma separated list of GPU(s) to use.')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    print("main")
    args = get_arguments()
    hp.set_hparam_yaml(args.case2)
    logdir_train1 = '{}/{}/train1'.format(hp.logdir_path, args.case1)
    logdir_train2 = '{}/{}/train2'.format(hp.logdir_path, args.case2)

    if os.path.exists('./converted') == False:
        os.makedirs('./converted')

    if os.path.exists('./converted/debug') == False:
        os.makedirs('./converted/debug')
    
    print('case1: {}, case2: {}, logdir1: {}, logdir2: {}'.format(args.case1, args.case2, logdir_train1, logdir_train2))
    do_convert(args, logdir1=logdir_train1, logdir2=logdir_train2)

