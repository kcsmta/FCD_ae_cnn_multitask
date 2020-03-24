import os

'''Train a recurrent convolutional network on the IMDB sentiment
classification task.

Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
'''
# from __future__ import print_function
#
import keras.backend.tensorflow_backend as K
import os

import numpy as np
import sys

from keras import losses
from keras.engine import Model
from keras.layers.merge import concatenate,maximum,average,add
from keras.utils import np_utils

from keras.layers import Input, UpSampling1D, Flatten
from keras.layers.pooling import MaxPooling1D
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint

import Data_IO

nb_classes = 5
batch_size= 50

from keras.models import load_model


file_wordvec = '../asmdata/vec_embedding_no_ops.txt'


print('Loading data...\n')
print ('Load token-vec: '+ file_wordvec)
X_train =[]
X_CV = []
X_test=[]
y_train=[]
y_CV =[]
y_test=[]

# prob = 'SUBINC'
prob = 'SUMTRIAN'

model_path = './models/ae_models'
model_name = 'ae_' + prob + '.h5'

file_train = '../asmdata/'+prob+'_Seq_train.txt'
file_CV = '../asmdata/'+ prob + '_Seq_CV.txt'
file_test = '../asmdata/' + prob + '_Seq_test.txt'

# file_train = "../asmdata/debug_Seq_CV.txt"
# file_train = "../asmdata/debug_Seq_CV.txt"
# file_train = "../asmdata/debug_Seq_CV.txt"

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
is_padded = 0
if maxlen % 2 == 1:
    maxlen = maxlen + 1
    is_padded = 0
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


model = load_model(os.path.join(model_path, model_name))
latent_map_layer = model.get_layer('latent_map').output
global_pooling = GlobalMaxPooling1D()(latent_map_layer)
latent_map_model = Model(inputs=model.input, outputs=global_pooling)

latent_map_train = latent_map_model.predict(X_train, batch_size=batch_size)
latent_map_test = latent_map_model.predict(X_test, batch_size=batch_size)
latent_map_val = latent_map_model.predict(X_CV, batch_size=batch_size)

model.summary()

# print(type(latent_map))
# for map in latent_map:
#     print(type(map))
#     print(map.shape)
#     print(map)

# For speed of computation, only run on a subset
label_train = np.argmax(y_train, axis=1)
label_test = np.argmax(y_test, axis=1)
label_val = np.argmax(y_CV, axis=1)

for i in range(len(latent_map_train)-is_padded):
    with open(prob+'_Embedding_train.txt', 'a') as f:
        text_line = str(label_train[i]) + " " + " ".join(map(str, latent_map_train[i])) + "\n"
        f.write(text_line)

for i in range(len(latent_map_val)-is_padded):
    with open(prob+'_Embedding_CV.txt', 'a') as f:
        text_line = str(label_val[i]) + " " + " ".join(map(str, latent_map_val[i])) + "\n"
        f.write(text_line)

for i in range(len(latent_map_test)-is_padded):
    with open(prob+'_Embedding_test.txt', 'a') as f:
        text_line = str(label_test[i]) + " " + " ".join(map(str, latent_map_test[i])) + "\n"
        f.write(text_line)
