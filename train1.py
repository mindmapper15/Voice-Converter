# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function

import argparse
import multiprocessing
import os
import warnings

from tensorpack.callbacks.saver import ModelSaver
from tensorpack.callbacks import InferenceRunner, ScalarStats
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.train import AutoResumeTrainConfig
from tensorpack.train.interface import launch_train_with_config
from tensorpack.train.trainers import SimpleTrainer
from tensorpack.train.trainers import SyncMultiGPUTrainerReplicated
from tensorpack.utils import logger
from tensorpack.input_source.input_source import QueueInput
from data_load import Net1DataFlow
from hparam import hparam as hp
from models import Net1
from preprocdata1 import preprocessing
import tensorflow as tf


def train(args, logdir):
    # model
    model = Net1()

    # dataflow
    TIMIT_TRAIN_WAV = 'TIMIT/TRAIN/*/*/*.npz'
    TIMIT_TEST_WAV = 'TIMIT/TEST/*/*/*.npz'

    print(os.path.join(hp.train1.preproc_data_path, args.case, TIMIT_TRAIN_WAV))
    print(os.path.join(hp.train1.preproc_data_path, args.case, TIMIT_TEST_WAV))

    df = Net1DataFlow(os.path.join(hp.train1.preproc_data_path, args.case, TIMIT_TRAIN_WAV), hp.train1.batch_size)
    df_test = Net1DataFlow(os.path.join(hp.train1.preproc_data_path, args.case, TIMIT_TEST_WAV), hp.train1.batch_size)

    # set logger for event and model saver
    logger.set_logger_dir(logdir)
    train_conf = AutoResumeTrainConfig(
        model=model,
        data=QueueInput(df(n_prefetch=1000, n_thread=8)),
        callbacks=[
            ModelSaver(checkpoint_dir=logdir),
            InferenceRunner(df_test(n_prefetch=1),
                            ScalarStats(['net1/eval/loss', 'net1/eval/acc'],prefix='')),
        ],
        max_epoch=hp.train1.num_epochs,
        steps_per_epoch=hp.train1.steps_per_epoch,
        #session_config=session_conf
    )
    ckpt = '{}/{}'.format(logdir, args.ckpt) if args.ckpt else tf.train.latest_checkpoint(logdir)

    if ckpt:
        train_conf.session_init = SaverRestore(ckpt)

    if hp.default.use_gpu == True:
        os.environ['CUDA_VISIBLE_DEVICES'] = hp.default.gpu_list
        train_conf.nr_tower = len(hp.default.gpu_list.split(','))
        num_gpu = len(hp.default.gpu_list.split(','))
        trainer = SyncMultiGPUTrainerReplicated(num_gpu)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        trainer = SimpleTrainer()

    launch_train_with_config(train_conf, trainer=trainer)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='experiment case name')
    parser.add_argument('-ckpt', help='checkpoint to load model.')
    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':
    from shutil import copyfile
    from glob import glob
    warnings.simplefilter(action='ignore', category=FutureWarning)

    args = get_arguments()
    hp.set_hparam_yaml(args.case)

    casedir = '{}'.format(hp.logdir)
    logdir_train1 = '{}/train1'.format(hp.logdir)

    if len(glob(os.path.join(casedir, 'default.yaml'))) == 0:
        if os.path.isdir(casedir) == False:
            os.makedirs(casedir)
        copyfile('hparams/default.yaml', os.path.join(casedir, 'default.yaml'))
    else:
        hp.set_hparam_yaml(case=args.case, default_file=os.path.join(casedir, 'default.yaml'))

    print('case: {}, logdir: {}, casedir: {}, Data Path: {}'.format(args.case, logdir_train1, casedir, hp.train1.dataset_path))

    preprocessing(hp.train1.dataset_path, os.path.join(hp.train1.preproc_data_path, args.case))
    train(args, logdir=logdir_train1)

    print("Done")
