# -*- coding: utf-8 -*-
# /usr/bin/python2


from __future__ import print_function

import argparse
import os
import glob
import matplotlib.pyplot as plt

from models import Net2ForConvert
import numpy as np
from audio import inv_preemphasis, denormalize_db, f0_adapt, preemphasis
import datetime
import tensorflow as tf
import librosa
import pyworld as pw
from hparam import hparam as hp
from data_load import _get_mfcc, _get_spectral_envelope

from tensorflow.python.client import timeline
from tensorpack.predict.base import OfflinePredictor
from tensorpack.predict.config import PredictConfig
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.tfutils.sessinit import ChainInit
from tensorpack.callbacks.base import Callback

class ConvertCallback(Callback):
    def __init__(self, logdir, test_per_epoch=1):
        self.df = Net2DataFlow(hp.convert.data_path, hp.convert.batch_size)
        self.logdir = logdir
        self.test_per_epoch = test_per_epoch

    def _setup_graph(self):
        tf.reset_default_graph()
        tf.Graph().as_default()
        self.predictor = self.trainer.get_predictor(
            get_eval_input_names(),
            get_eval_output_names())

    def _trigger_epoch(self):
        if self.epoch_num % self.test_per_epoch == 0:
            pred_spec_env, y_spec_env, _ = convert_spectral_envelope(self.predictor, self.mfcc, self.)
            pred_spec_env = np.expand_dims(pred_spec_env, 3)
            y_spec_env = np.expand_dims(y_spec_env, 3)
            self.trainer.monitors.put_scalar('eval/loss', loss)

            #Write the result
            tf.summary.image('Predicted Spectral Envelope', pred_spec_env, max_outputs=1)
            tf.summary.image('Original Spectral Envelope', y_spec_env, max_outputs=1)

def convert_spectral_envelope(predictor, mfcc, sp_en):
    pred_s = datetime.datetime.now()
    pred_sp_en, _, ppgs = predictor(mfcc, sp_en)

    pred_e = datetime.datetime.now()
    pred_t = pred_e - pred_s
    print("Predicting time:{}s".format(pred_t.seconds))

    return pred_sp_en, ppgs


def get_eval_input_names():
    return ['x_mfccs', 'y_sp_en']


def get_eval_output_names():
    return ['pred_sp_en', 'y_sp_en', 'ppgs']


def set_enviroment(args, logdir1, logdir2):
    # Load graph
    set_env_s = datetime.datetime.now()

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
    predictor = OfflinePredictor(pred_conf)

    set_env_e = datetime.datetime.now()
    set_env_t = set_env_e - set_env_s
    print("Setting Environment time:{}s".format(set_env_t.seconds))

    return predictor

def do_convert(predictor, input_name, logdir2):
    convert_s = datetime.datetime.now()

    # Load input audio
    input_audio, _ = librosa.load(input_name, sr=hp.default.sr, dtype=np.float64)

    # Extract F0 from input audio first
    input_f0, t_table = pw.dio(input_audio, hp.default.sr)
    input_f0 = pw.stonemask(input_audio, input_f0, t_table, hp.default.sr)

    # Get MFCC, Spectral Envelope, and Aperiodicity
    mfcc = _get_mfcc(input_audio, hp.default.n_fft, hp.default.win_length, hp.default.hop_length)
    mfcc = np.expand_dims(mfcc, axis=0)

    input_ap = pw.d4c(input_audio, input_f0, t_table, hp.default.sr, fft_size=hp.default.n_fft)

    input_sp_en = _get_spectral_envelope(preemphasis(input_audio, coeff=hp.default.preemphasis), hp.default.n_fft)
    plt.imsave('./converted/debug/input_sp_en_original.png', input_sp_en, cmap='binary')
    input_sp_en = np.expand_dims(input_sp_en, axis=0)

    # Convert Spectral Envelope
    output_sp_en, ppgs = convert_spectral_envelope(predictor, mfcc, input_sp_en)
    output_sp_en = np.squeeze(output_sp_en.astype(np.float64), axis=0)

    preproc_s = datetime.datetime.now()
    # Denormalization
    output_sp_en = denormalize_db(output_sp_en, hp.default.max_db, hp.default.min_db)

    # Db to amp
    output_sp_en = librosa.db_to_amplitude(output_sp_en)

    # Emphasize the magnitude
    output_sp_en = np.power(output_sp_en, hp.convert.emphasis_magnitude)

    preproc_e = datetime.datetime.now()
    preproc_t = preproc_e - preproc_s
    print("Pre-Processing time:{}s".format(preproc_t.seconds))

    # F0 transformation with WORLD Vocoder
    output_f0 = f0_adapt(input_f0, logdir2)

    # Synthesize audio and de-emphasize
    output_audio = pw.synthesize(output_f0, output_sp_en, input_ap, hp.default.sr)
    output_audio = inv_preemphasis(output_audio, coeff=hp.default.preemphasis)

    # Saving output_audio to 32-bit Float wav file
    output_audio = output_audio.astype(np.float32)
    librosa.output.write_wav(path="./converted/"+input_name,y=output_audio,sr=hp.default.sr)

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
    predictor = set_enviroment(args, logdir_train1, logdir_train2)

    while True:
        input_name = input("Write your audio file\'s path for converting : ")
        if input_name == 'quit':
            break
        elif len(glob.glob(input_name)) == 0:
            print("That audio file doesn't exist! Try something else.")
            continue
        else:
            do_convert(predictor, input_name, logdir_train2)
