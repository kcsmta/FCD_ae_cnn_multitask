'''Train a recurrent convolutional network on the IMDB sentiment
classification task.

Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
'''
# from __future__ import print_function
#
import keras.backend.tensorflow_backend as K
import os
import tensorflow as tf

tf_config = K.tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = K.tf.compat.v1.Session(config=tf_config)
# K.set_session(session)
tf.compat.v1.keras.backend.set_session(session)

import numpy as np
import sys

from keras import losses
from keras.engine import Model
from keras.layers.merge import concatenate,maximum,average,add
from keras.utils import np_utils

from keras.layers import Input, UpSampling1D
from keras.layers.pooling import MaxPooling1D
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint

import Data_IO

import params

import plot_training_process

nb_classes = params.nb_classes #number of classes
embddingsize = params.embddingsize # the length of word vectors
max_len=params.max_len # max length of a sentence

#convolution w1, w2, ... --> pooling ---> merge ---> dense ---> output
# Convolution
filter_length = params.filter_length # the size of filters
conv_filters = params.conv_filters # number of filters for each conv layer
pool_size = params.pool_size

num_hid = params.num_hid

# activation function
activation =params.activation
# Training
batch_size = params.batch_size

model_path = params.model_path
results_path = params.results_path
data_path= params.data_path
data_part = params.data_part

def run_net(probs, nb_epoch = params.nb_epoch):
    # probs = problems MNMX, FLOW16, ...
    # pretrained_name = pretrained model, if None then training from scratch
    # finetune = continue training from pretrained model
    # model = name of the model after training
    # input data
    # prob = sys.argv[1]#'MNMX'

    file_wordvec = '../asmdata/vec_embedding_no_ops.txt'


    print('Loading data...\n')
    print ('Load token-vec: '+ file_wordvec)
    X_train =[]
    X_CV = []
    X_test=[]
    y_train=[]
    y_CV =[]
    y_test=[]

    # print (word2id)
    for prob in probs:
        file_train = '../asmdata/' + prob + '_Seq_train.txt'
        file_CV = '../asmdata/' + prob + '_Seq_CV.txt'
        file_test = '../asmdata/' + prob + '_Seq_test.txt'

        print('\nLoad training data: ' + file_train)
        print('\nLoad CV data: ' + file_CV)
        print('\nLoad test data: ' + file_test)

        wordvec = Data_IO.loadWordEmbedding(file_wordvec)
        y_train, X_train, maxlen_train = Data_IO.load_ASMSeqData(file_train, wordvec)
        y_CV, X_CV, maxlen_CV = Data_IO.load_ASMSeqData(file_CV, wordvec)
        y_test, X_test, maxlen_test = Data_IO.load_ASMSeqData(file_test, wordvec)

        if nb_classes == 2:
            y_train = [x if x == 0 else 1 for x in y_train]
            y_CV = [x if x == 0 else 1 for x in y_CV]
            y_test = [x if x == 0 else 1 for x in y_test]

        y_testnum = y_test

        # maxlen: the length of the longest instruction sequence
        maxlen = np.max([maxlen_train, maxlen_CV, maxlen_test])
        if maxlen % 2 == 1:
            maxlen = maxlen + 1
        print('max number of instructions: ' + str(maxlen))
        # padding data
        Data_IO.paddingASMSeq(X_train, maxlen)
        Data_IO.paddingASMSeq(X_CV, maxlen)
        Data_IO.paddingASMSeq(X_test, maxlen)

        X_train = np.array(X_train)
        X_CV = np.array(X_CV)
        X_test = np.array(X_test)

        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_CV = np_utils.to_categorical(y_CV, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)

        print('num train :' + str(len(X_train)))
        print('num CV :' + str(len(X_CV)))
        print('num test :' + str(len(X_test)))
        print('Build model...')


    input = Input(shape=((maxlen, embddingsize)), dtype='float32')
    x = Conv1D(filters=conv_filters[0], kernel_size=filter_length, strides=1, padding='same', dilation_rate=1,
               use_bias=True, activation=activation)(input)
    print('filter num =' + str(conv_filters[0]))
    for conv in range(1, len(conv_filters)):
        print('filter num =' + str(conv_filters[conv]))
        x = MaxPooling1D(pool_size=pool_size, strides=None, padding='same')(x)
        if conv == len(conv_filters) - 1:
            x = Conv1D(filters=conv_filters[conv], kernel_size=filter_length, strides=1, padding='same',
                       dilation_rate=1, use_bias=True, activation=activation, name='latent_map')(x)
        else:
            x = Conv1D(filters=conv_filters[conv], kernel_size=filter_length, strides=1, padding='same',
                       dilation_rate=1, use_bias=True, activation=activation)(x)
    encode = x
    # x = GlobalMaxPooling1D()(x)
    # x = Dense(num_hid, activation=activation)(x)
    # x = Dense(nb_classes, activation='softmax')(x)
    conv_filters_decoder = conv_filters
    conv_filters_decoder.reverse()

    for conv in range(1, len(conv_filters_decoder)):
        print('filter num =' + str(conv_filters_decoder[conv]))
        x = Conv1D(filters=conv_filters[conv], kernel_size=filter_length, strides=1, padding='same',
                   dilation_rate=1, use_bias=True, activation=activation)(x)
        x = UpSampling1D(size=pool_size)(x)

    # decode equivalent to embedding layer
    x = Conv1D(filters=embddingsize, kernel_size=filter_length, strides=1, padding='same',
               dilation_rate=1, use_bias=True, activation=activation, name='decode_layer')(x)

    model = Model(inputs=input, outputs=x)
    encoder_model = Model(inputs=input, output=encode)
    print(model.summary())

    #loss = 'mean_squared_error', 'binary_crossentropy','categorical_crossentropy'
    from keras import optimizers
    sgd = optimizers.SGD(lr=0.4, clipnorm=1.)
    model.compile(loss= 'mean_squared_error',#'mean_squared_error', #binary_crossentropy
                  optimizer= sgd,
                  metrics=['accuracy'])

    print('Training...')
    hist = model.fit(X_train, X_train, batch_size=batch_size, epochs=nb_epoch, verbose=True, validation_data=(X_CV, X_CV))
    plot_training_process.plot_training_process(hist.history, results_path, prob, data_part, net="ae")
    # save model
    model.save(os.path.join(model_path, "ae_" + prob + ".h5"))
    encoder_model.save(os.path.join(model_path, "ae_encoder_" + prob + ".h5"))

