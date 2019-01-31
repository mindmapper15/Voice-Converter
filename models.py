# -*- coding: utf-8 -*-
# !/usr/bin/env python

import tensorflow as tf
from tensorpack.graph_builder.model_desc import ModelDesc, InputDesc
from tensorpack.tfutils import (
    get_current_tower_context, optimizer, gradproc)
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

import tensorpack_extension
from data_load import phns
from hparam import hparam as hp
from modules import prenet, cbhg, normalize


class Net1(ModelDesc):
    def __init__(self):
        pass

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, None, hp.default.n_mfcc), 'x_mfccs'),
                InputDesc(tf.int32, (None, None,), 'y_ppgs')]

    def _build_graph(self, inputs):
        self.x_mfccs, self.y_ppgs = inputs
        is_training = get_current_tower_context().is_training
        with tf.variable_scope('net1'):
            self.ppgs, self.preds, self.logits = self.network(self.x_mfccs, is_training)
        self.cost = self.loss()
        acc = self.acc()

        # summaries
        tf.summary.scalar('net1/train/loss', self.cost)
        tf.summary.scalar('net1/train/acc', acc)

        if not is_training:
            # summaries
            tf.summary.scalar('net1/eval/summ_loss', self.cost)
            tf.summary.scalar('net1/eval/summ_acc', acc)

            # for confusion matrix
            tf.reshape(self.y_ppgs, shape=(tf.size(self.y_ppgs),), name='net1/eval/y_ppg_1d')
            tf.reshape(self.preds, shape=(tf.size(self.preds),), name='net1/eval/pred_ppg_1d')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=hp.train1.lr, trainable=False)
        return tf.train.AdamOptimizer(lr)

    @auto_reuse_variable_scope
    def network(self, x_mfcc, is_training):
        # Pre-net
        prenet_out = prenet(x_mfcc,
                            num_units=[hp.train1.hidden_units, hp.train1.hidden_units // 2],
                            dropout_rate=hp.train1.dropout_rate,
                            is_training=is_training)  # (N, T, E/2)

        # CBHG
        out = cbhg(prenet_out, hp.train1.num_banks, hp.train1.hidden_units // 2,
                   hp.train1.num_highway_blocks, hp.train1.norm_type, is_training)

        # Final linear projection
        logits = tf.layers.dense(out, len(phns))  # (N, T, V)
        ppgs = tf.nn.softmax(logits / hp.train1.t, name='ppgs')  # (N, T, V)
        preds = tf.to_int32(tf.argmax(logits, axis=-1))  # (N, T)

        return ppgs, preds, logits

    def loss(self):
        istarget = tf.sign(tf.abs(tf.reduce_sum(self.x_mfccs, -1)))  # indicator: (N, T)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits / hp.train1.t,
                                                              labels=self.y_ppgs)
        loss *= istarget
        loss = tf.reduce_mean(loss)
        return loss

    def acc(self):
        istarget = tf.sign(tf.abs(tf.reduce_sum(self.x_mfccs, -1)))  # indicator: (N, T)
        num_hits = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y_ppgs)) * istarget)
        num_targets = tf.reduce_sum(istarget)
        acc = num_hits / num_targets
        return acc


class Net2(ModelDesc):

    def _get_inputs(self):
        n_timesteps = (hp.default.duration * hp.default.sr) // hp.default.hop_length + 1
        # timestep의 갯수, 전체 길이를 hop length 만큼 나눈 것.
        # STFT 변환 시, 전체 오디오에서 생성되는 총 window 개수와 동일

        # 그래프에 입력하는 entry point에 관한 메타데이터를 생성한다.
        # 이러한 메타데이터는 나중에 placeholder을 만들거나
        # 다른 타입의 입력값을 만드는데 쓰일 수 있다.
        # placeholder : 데이터의 형태만 지정한 뒤, 실제 입력은 실행 단계에서 받을 수 있는 텐서를 뜻한다.
        return [InputDesc(tf.float32, (None, n_timesteps, hp.default.n_mfcc), 'x_mfccs'),
                InputDesc(tf.float32, (None, n_timesteps, hp.default.n_fft // 2 + 1), 'y_sp_en'),]

    def _build_graph(self, inputs):
        self.x_mfcc, self.y_sp_en = inputs

        is_training = get_current_tower_context().is_training

        # build net1
        # train1에서 학습된 SI-ASR 모델에서 PPGs 추출.
        # 목표 음성에서 추출된 MFCC를 입력해 해당 MFCC에 대한 PPG들을 뽑아낸다.
        self.net1 = Net1()
        with tf.variable_scope('net1'):
            self.ppgs, _, _ = self.net1.network(self.x_mfcc, is_training)
        self.ppgs = tf.identity(self.ppgs, name='ppgs')

        # build net2
        # net1을 통과시켜 얻은 PPGs를 이용해 스펙트로그램과 mel-스펙트로그램을 예측한다.
        with tf.variable_scope('net2'):
            self.pred_sp_en = self.network(self.ppgs, is_training)
        self.pred_sp_en = tf.identity(self.pred_sp_en, name='pred_sp_en')

        self.cost = self.loss()
        diff = self.diff()

        # summaries
        tf.summary.scalar('net2/train/loss', self.cost)
        tf.summary.scalar('net1/train/acc', diff)

        if not is_training:
            tf.summary.scalar('net2/eval/summ_loss', self.cost)
            tf.summary.scalar('net1/train/acc', diff)

    def _get_optimizer(self):
        gradprocs = [
            tensorpack_extension.FilterGradientVariables('.*net2.*', verbose=False),
            gradproc.MapGradient(
                lambda grad: tf.clip_by_value(grad, hp.train2.clip_value_min, hp.train2.clip_value_max)),
            gradproc.GlobalNormClip(hp.train2.clip_norm),
            # gradproc.PrintGradient(),
            # gradproc.CheckGradient(),
        ]
        lr = tf.get_variable('learning_rate', initializer=hp.train2.lr, trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        return optimizer.apply_grad_processors(opt, gradprocs)

    @auto_reuse_variable_scope
    def network(self, ppgs, is_training):
        # Pre-net
        # Net1에서 예측한 PPGs를 pre-network에 삽입하여 prenet_out을 얻는다.
        prenet_out = prenet(ppgs,
                            num_units=[hp.train2.hidden_units, hp.train2.hidden_units // 2],
                            dropout_rate=hp.train2.dropout_rate,
                            is_training=is_training)  # (N, T, E/2)

        # CBHG: spectral_envelope
        # prenet_out을 CBHG에 넣어
        pred_sp_en = cbhg(prenet_out, hp.train2.num_banks, hp.train2.hidden_units // 2,
                        hp.train2.num_highway_blocks, hp.train2.norm_type, is_training,
                        scope="cbhg_sp_en")
        pred_sp_en = tf.layers.dense(pred_sp_en, self.y_sp_en.shape[-1], name='pred_sp_en')  # (N, T, n_mels)

        return pred_sp_en

    def loss(self):
        # 목표 음성에서 얻은 스펙트로그램, mel-스펙트로그램과
        # 예측을 통해 얻은 스펙트로그램, mel-스펙트로그램을 서로 비교하여 loss를 계산
        return tf.reduce_mean(tf.squared_difference(self.pred_sp_en, self.y_sp_en))


class Net2ForConvert(Net2):

    def _get_inputs(self):

        return [InputDesc(tf.float32, (None, None, hp.default.n_mfcc), 'x_mfccs'),
                InputDesc(tf.float32, (None, None, hp.default.n_fft // 2 + 1), 'y_sp_en'), ]
