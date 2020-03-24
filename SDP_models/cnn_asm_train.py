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

from keras.layers import Input, Dropout
from keras.layers.pooling import MaxPooling1D
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import accuracy_score, f1_score

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

        train_data = int(data_part*len(X_train))

        X_train = np.array(X_train)
        X_train = X_train[:train_data]
        y_train = y_train[:train_data]
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
    # embedding = Embedding(input_dim=len(word2id), output_dim=embddingsize, input_length=max_len)(input)

    # embedding = Embedding(len(word2id), embddingsize, weights=[embedding_matrix],
    #                       input_length=max_len, trainable=False)(input)

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
    x = GlobalMaxPooling1D()(x)
    x = Dense(num_hid, activation=activation)(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes, activation='softmax', name='predict_layer')(x)

    model = Model(inputs=input, outputs=x)

    print(model.summary())

    #loss = 'mean_squared_error', 'binary_crossentropy','categorical_crossentropy'
    from keras import optimizers
    sgd = optimizers.SGD(lr=0.4, clipnorm=1.)
    model.compile(loss= losses.categorical_crossentropy,#'mean_squared_error', #binary_crossentropy
                  optimizer= sgd,
                  metrics=['accuracy'])

    print('Training...')
    best_weights = ModelCheckpoint('best_cnn.h5', verbose=1, monitor='val_accuracy', save_best_only=True, mode='auto')
    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=True, validation_data=(X_CV, y_CV),
                     callbacks=[best_weights])
    plot_training_process.plot_training_process(hist.history, results_path, prob, data_part, net="cnn")
    # save model
    # model.save(os.path.join(model_path, "cnn_" + prob + "_" + str(int(data_part*100)) + ".h5"))
    import shutil
    shutil.copyfile('best_cnn.h5', os.path.join(model_path, "cnn_" + prob + "_" + str(int(data_part * 100)) + ".h5"))

    # # save history to json file
    # import json
    # with open('P%s_M%s.json'%(pretrained_name, model_name), 'wb') as f:
    #     json.dump(hist.history, f)
    # save history to pickle file
    # import pickle
    # with open('cnn_' + prob + "_" + str(int(data_part*100)) + '.pkl', 'wb') as f:
    #     pickle.dump(hist.history, f)


    model = load_model(os.path.join(model_path, "cnn_" + prob + "_" + str(int(data_part * 100)) + ".h5"))
    L = model.predict(X_test, batch_size=batch_size)

    print(L.shape)
    print(y_test.shape)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(L, axis=1)

    acc = accuracy_score(y_true, y_pred)
    # f1_macro = f1_score(y_true, y_pred, average='macro')
    # f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print("Accuracy=", acc)
    # print("F1-macro=", f1_macro)
    # print("F1-micro=", f1_micro)
    print("F1-score=", f1_weighted)

    import operator
    count = 0

    fout = open(os.path.join(results_path, 'cnn_' + prob + "_" + str(int(data_part * 100)) + '.roc'), 'w')
    fout.write('label, ')
    label_list = np.unique(y_testnum)
    fout.write(', '.join([str(i) for i in label_list]))
    fout.write('\n')
    for idx, probs in enumerate(L):
        pred_label, value = max(enumerate(probs), key=operator.itemgetter(1))
        if pred_label == y_testnum[idx]:
            count += 1
        fout.write(str(y_testnum[idx]) + ' ')
        fout.write(' '.join([str(i) for i in probs]))
        fout.write('\n')
    print("ROC's saved to {}".format(
        os.path.join(results_path, 'cnn_' + prob + "_" + str(int(data_part * 100)) + '.roc')))
    fout.close()
