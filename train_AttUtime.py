decompcnn# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 11:04:47 2022
@author: mathilde letard
"""
from __future__ import print_function, division
import scipy
import keras
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import h5py
from keras import backend as K
from tensorflow.keras.layers import add, Input, Conv1D, BatchNormalization, Activation, MaxPool1D, UpSampling1D, Concatenate, Multiply
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import schedules, Adam
from dataloader_deconv import DataLoader
from default_deconv import *


def custom_loss(coef_y1=2.,coef_y2=2., coef_y3=3., coef_global=5., coef_grad=5.):
    def deconv_loss(y_true, y_pred):
        mae = tf.reduce_mean(tf.math.abs((y_true - y_pred) * (1+y_true)))
        dist_pred = y_pred / tf.expand_dims(tf.math.reduce_sum(y_pred, axis=1), axis=-1)
        dist_true = y_true / tf.expand_dims(tf.math.reduce_sum(y_true, axis=1), axis=-1)
        bat = tf.reduce_sum(tf.math.sqrt(tf.math.multiply(dist_true[...,0],dist_pred[...,0])), axis=-1)
        bat = tf.reduce_mean(bat)
        bat = 1-bat
        return mae + bat
    return deconv_loss


class decomposition_network():
    def __init__(self,
                 dataset_train,
                 dataset_val,
                 wf_len=DEF_LEN,
                 channels_out=DEF_CH_OUT,
                 channels_in=DEF_CH_IN,
                 filepath_save=DEF_PATH_SAVE,
                 initial_lr=DEF_INITIAL_LR,
                 decay_rate=DEF_DECAY_RATE,
                 decay_steps=DEF_DECAY_STEPS,
                 gf=DEF_FILTERS):
        self.dataset_train=dataset_train
        self.dataset_val=dataset_val
        self.wf_len = wf_len
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.filepath_save = filepath_save
        self.train_loader=DataLoader(self.dataset_train)
        self.val_loader=DataLoader(self.dataset_val)
        self.gf = gf
        self.name_readme='%s/%s'%(self.filepath_save,DEF_README)
        self.cost=[]
        self.valid=[]
        self.epo=[]
        fid = open(self.name_readme, "a")
        fid.write('--------------------\n')
        fid.write('Network Optimization\n')
        fid.write('--------------------\n')
        fid.write('Initial learning rate : %.6f\n' % self.initial_lr)
        fid.write('Decay rate : %.6f\n' % self.decay_rate)
        fid.write('Decay steps : %d\n' % self.decay_steps)
        fid.close()

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_lr,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate)

        self.decompcnn = self.multioutput_AttentionUnet1D()
        self.decompcnn.compile(loss={"surface": custom_loss(), "column":custom_loss(), "bottom":custom_loss()},
                               optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                               metrics={"surface": custom_loss(), "column":custom_loss(), "bottom":custom_loss()})


    def multioutput_AttentionUnet1D(self):
        def conv_block(inputs, filters, pool=True):
            x = Conv1D(filters, 3, padding="same")(inputs)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv1D(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            if pool == True:
                p = MaxPool1D(2)(x)
                return x, p
            else:
                return x

        def attention_block(query, inputs, filters, n):
            x1 = Conv1D(filters, 1, padding="same")(query)
            x11 = Conv1D(filters, 1, padding="same")(inputs)
            x1 = UpSampling1D(2)(x1)
            a = add([x1, x11])
            a = Activation("relu")(a)
            a = Conv1D(1, 1, padding="same", activation="sigmoid",name=n)(a)
            return Multiply()([inputs, a])

        waveform = Input((self.wf_len, self.channels_in))
        """ Encoder """
        x1, p1 = conv_block(waveform, self.gf, pool=True)
        x2, p2 = conv_block(p1, self.gf*2, pool=True)
        x3, p3 = conv_block(p2, self.gf*3, pool=True)
        x4, p4 = conv_block(p3, self.gf*4, pool=True)
        """ Bridge """
        b1 = conv_block(p4, self.gf*8, pool=False)
        """ Decoder """
        a1 = attention_block(b1, x4, self.gf*8, 'att_1')
        u1 = UpSampling1D(2)(b1)
        c1 = Concatenate()([u1, a1])
        x5 = conv_block(c1, 4*self.gf, pool=False)
        a2 = attention_block(x5, x3, 4*self.gf, 'att_2')
        u2 = UpSampling1D(2)(x5)
        c2 = Concatenate()([u2, a2])
        x6 = conv_block(c2, 3*self.gf, pool=False)
        a3 = attention_block(x6, x2, 3*self.gf, 'att_3')
        u3 = UpSampling1D(2)(x6)
        c3 = Concatenate()([u3, a3])
        x7 = conv_block(c3, self.gf*2, pool=False)
        a4 = attention_block(x7, x1, self.gf*2, 'att_4')
        u4 = UpSampling1D(2)(x7)
        c4 = Concatenate()([u4, a4])
        x8 = conv_block(c4, self.gf, pool=False)
        """ Output layer """
        surface = Conv1D(1, 1, padding="same", activation="sigmoid", name="surface")(x8)
        column = Conv1D(1, 1, padding="same", activation="sigmoid", name="column")(x8)
        bottom = Conv1D(1, 1, padding="same", activation="sigmoid", name="bottom")(x8)
        return Model(waveform, [surface,column,bottom])


    def train(self,epochs, batch_size):
        start_time = datetime.datetime.now()
        moy_loss = 0
        Niter_epoch=int(self.train_loader.nb_ech/batch_size)+1
        for iter in range(epochs*Niter_epoch):
            X, y = self.train_loader.load_batch(batch_size)
            g_loss = self.decompcnn.train_on_batch(X, y)
            elapsed_time = datetime.datetime.now() - start_time
            g_loss, g_loss1, g_loss2, g_loss3, _, __, ___ =g_loss
            toprint = "[%.7d] [time: %s] --gen losses: surface %f - column %f - bottom %f " % (iter, elapsed_time, g_loss1, g_loss2, g_loss3)
            sys.stdout.write(toprint + chr(13))
            mean_g_loss = tf.reduce_mean([g_loss1, g_loss2, g_loss3])
            moy_loss = moy_loss + mean_g_loss / Niter_epoch
            if iter % Niter_epoch == 0 and iter > 3:
                print('\n')
                self.chek_training(iter//Niter_epoch, moy_loss)
                moy_loss = 0
        self.dataset_train.close()
        self.dataset_val.close()

    def chek_training(self,iter_train,gloss=0):
        self.epo.append(iter_train)
        lossf=0.
        moy_valid_loss = 0
        Niter=int(self.val_loader.nb_ech/batch_size)
        init = False
        Ypredict1 = 0
        Ypredict2 = 0
        Ypredict3 = 0
        Yval1 = 0
        Yval2 = 0
        Yval3 = 0
        Xval = 0
        for iter in range(Niter):
            Xval_batch, Yval_batch = self.val_loader.load_val_batch(batch_size,iter)
            Yval1_batch = tf.cast(Yval_batch[0], tf.float32)
            Yval2_batch = tf.cast(Yval_batch[1], tf.float32)
            Yval3_batch = tf.cast(Yval_batch[2], tf.float32)
            if not init:
                Ypredict1, Ypredict2, Ypredict3 = self.decompcnn(Xval_batch)
                Xval = Xval_batch
                Yval1 = Yval1_batch
                Yval2 = Yval2_batch
                Yval3 = Yval3_batch
                init = True
                valid_loss1=custom_loss()(Yval1_batch,Ypredict1)
                valid_loss2=custom_loss()(Yval2_batch,Ypredict2)
                valid_loss3=custom_loss()(Yval3_batch,Ypredict3)
                eval_valid_loss=tf.reduce_mean([valid_loss1,valid_loss2,valid_loss3])
            else:
                pred_batch = self.decompcnn(Xval_batch)
                Ypredict1 = np.vstack((Ypredict1,pred_batch[0]))
                Ypredict2 = np.vstack((Ypredict2,pred_batch[1]))
                Ypredict3 = np.vstack((Ypredict3,pred_batch[2]))
                Yval1 = np.vstack((Yval1,Yval1_batch))
                Yval2 = np.vstack((Yval2,Yval2_batch))
                Yval3 = np.vstack((Yval3,Yval3_batch))
                Xval = np.vstack((Xval,Xval_batch))
                valid_loss1=custom_loss()(Yval1_batch,pred_batch[0])
                valid_loss2=custom_loss()(Yval2_batch,pred_batch[1])
                valid_loss3=custom_loss()(Yval3_batch,pred_batch[2])
                eval_valid_loss=tf.reduce_mean([valid_loss1,valid_loss2,valid_loss3])
            moy_valid_loss = moy_valid_loss + eval_valid_loss / Niter
        self.cost.append(gloss)
        self.valid.append(moy_valid_loss)
        fig=plt.plot(self.epo,self.cost)
        fig=plt.plot(self.epo,self.valid)
        fig=plt.legend(['train_loss','valid_loss'])
        name_fig='%s/graph.png'%(self.filepath_save)
        fig.figure.savefig(name_fig)
        fig=plt.close()
        print('Epoch : %.8d -- Av generator loss : %f ; Av Valid  loss : %f '%(iter_train,gloss, moy_valid_loss))
        self.decompcnn.save('%s/models/model_%.8d_loss_%.5f_val_%.5f.h5'%(self.filepath_save, iter_train,gloss,moy_valid_loss))
        self.plot_res(Ypredict1, Ypredict2, Ypredict3,Yval1, Yval2, Yval3 ,Xval,iter_train)


    def plot_res(self,Ypredict1, Ypredict2, Ypredict3,Yval1, Yval2, Yval3 ,Xval,iter):
        for i in range(0, Yval1.shape[0], 50):
            fig = plt.figure()
            fig, axs = plt.subplots(3)
            fig.tight_layout(pad=2)
            axs[0].plot(Xval[i], color='k')
            axs[1].plot(Yval1[i], color='royalblue', label='surface')
            axs[1].plot(Yval2[i], color='dimgray', label='column')
            axs[1].plot(Yval3[i], color='crimson',label='bottom')
            axs[2].plot(Ypredict1[i], color='royalblue', label='surface')
            axs[2].plot(Ypredict2[i], color='dimgray', label='column')
            axs[2].plot(Ypredict3[i], color='crimson', label='bottom')
            axs[0].set_ylim(0,1)
            axs[1].set_ylim(0,1)
            axs[2].set_ylim(0,1)
            axs[1].legend()
            axs[2].legend()
            axs[0].set_title('Initial waveform', fontsize=10)
            axs[1].set_title('True deconvolution', fontsize=10)
            axs[2].set_title('Predicted deconvolution', fontsize=10)
            name_fig='%s/valid/valid_%.3d_epoch_%.5d.png'%(self.filepath_save,i,iter)
            fig.savefig(name_fig)
            plt.close('all')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Waveform decomposition")
    parser.add_argument("--path_train", required=True,
                        help="path to h5 train file")
    parser.add_argument("--path_val", required=True,
                        help="path to h5 val file")
    parser.add_argument("--batch_size", type=int, default=DEF_BATCH_SIZE,
                        help="Size of the batch (default : %d)"%DEF_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEF_EPOCHS,
                        help="Number of epochs (default : %d)"%DEF_EPOCHS)
    parser.add_argument("--wflen", type=int, default=DEF_LEN,
                        help="numbers of columns of low res (default : %d)"%DEF_LEN)
    parser.add_argument("--ch_out", type=int, default=DEF_CH_OUT,
                        help="channels in output (default : %d)"%DEF_CH_OUT)
    parser.add_argument("--ch_in", type=int, default=DEF_CH_IN,
                        help="channels in input (default : %d)"%DEF_CH_IN)
    parser.add_argument("--gpu_ids", type=str, default=DEF_CUDA,
                        help="priority for GPU (default : %s)"%DEF_CUDA)
    parser.add_argument("--path_model",  type=str, default=DEF_PATH_SAVE,
                        help="path to save models ")
    parser.add_argument("--pretrain", type=str, default=DEF_PATH_PRETRAIN,
                        help="pretrained model (default : %s)"%DEF_PATH_PRETRAIN)
    parser.add_argument("--gf", type=int, default=DEF_FILTERS,
                        help="number of filter for the residual part (default : %d)"%DEF_FILTERS)
    parser.add_argument("--initial_lr", type=float, default=DEF_INITIAL_LR,
                        help="Initial learning rate (default : %f)"%DEF_INITIAL_LR)
    parser.add_argument("--decay_steps", type=int, default=DEF_DECAY_STEPS,
                        help="Decay steps (default : %d)"%DEF_DECAY_STEPS)
    parser.add_argument("--decay_rate", type=float, default=DEF_DECAY_RATE,
                        help="Decay rate (default : %f)"%DEF_DECAY_RATE)
    parser.add_argument("--note", type=str, default='',
                        help="Personal note for the readme file ")

    args = parser.parse_args()
    dataset_train=args.path_train
    dataset_val=args.path_val
    batch_size = args.batch_size
    epochs = int(args.epochs)
    wf_len=int(args.wflen)
    ch_out = int(args.ch_out)
    ch_in = int(args.ch_in)
    path_model = args.path_model
    cuda_id=args.gpu_ids
    gf= args.gf
    note=args.note
    decay_steps=args.decay_steps
    decay_rate=args.decay_rate
    initial_lr=args.initial_lr
    pretrain=args.pretrain

    if not os.path.exists(path_model):
        os.makedirs(path_model)
    if not os.path.exists('%s/valid'%path_model):
        os.makedirs('%s/valid'%path_model)
    if not os.path.exists('%s/models'%path_model):
        os.makedirs('%s/models'%path_model)
    if not os.path.exists('%s/python'%path_model):
        os.makedirs('%s/python'%path_model)
    commande='cp *.py %s/python'%path_model
    os.system(commande)
    name_readme='%s/%s'%(path_model,DEF_README)
    fid = open(name_readme, "w")
    fid.write('--------------------\n')
    fid.write('Input parameters\n')
    fid.write('Dataset train (h5)) : %s\n'%dataset_train)
    fid.write('Dataset val (h5)) : %s\n'%dataset_val)
    fid.write('Batch size : %d\n'%batch_size)
    fid.write('Epochs : %d\n'%epochs)
    fid.write('Size (len) : %d \n'%(wf_len))
    fid.write('Channels (in, out)  : %d x %d \n'%(ch_in,ch_out))
    fid.write('Number of filters  : %d \n'%(gf))
    fid.write('--------------------\n')
    fid.write('Losses\n')
    fid.write('--------------------\n')
    fid.write('--------------------\n')
    if pretrain is not '':
        fid.write('Pretrain\n')
        fid.write('--------------------\n')
        fid.write(pretrain)
        fid.write('--------------------\n')
    if note is not '':
        fid.write('Note : %s\n'%note)
        fid.write('--------------------\n')
    fid.write('--------------------\n')
    fid.write('Command\n')
    fid.write(' '.join(sys.argv))
    fid.write('\n')
    fid.write('--------------------\n')
    fid.close()
    net = decomposition_network(dataset_train=dataset_train,
                 dataset_val=dataset_val,
                 wf_len=wf_len,
                 channels_out=ch_out,
                 channels_in=ch_in,
                 filepath_save=path_model,
                 initial_lr=initial_lr,
                 decay_rate=decay_rate,
                 decay_steps=decay_steps,
                 gf=gf)
    if os.path.exists(pretrain):
        print('---------------------------------')
        print('load pretrain model %s'%pretrain)
        net.decompcnn=load_model(pretrain)
        print('Done')
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,beta_1=0.9,beta_2=0.999)
        self.decompcnn.compile(loss=tf.keras.losses.MeanSquaredError(),
                               optimizer=optimizer,
                               metrics=[tf.keras.losses.MeanSquaredError()])
        print('---------------------------------')
    else:
        print('---------------------------------')
        print('Train from scratch')
        print('---------------------------------')

    stringlist = []
    net.decompcnn.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    print(short_model_summary)
    fid = open(name_readme, "a")
    fid.write('--------------------\n')
    fid.write('model\n')
    fid.write(short_model_summary)
    fid.close()
    net.train(epochs=epochs, batch_size=batch_size)
    net.train_loader.close()
    net.val_loader.close()
