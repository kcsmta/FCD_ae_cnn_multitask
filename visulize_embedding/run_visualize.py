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

from keras.engine import Model
from keras.utils import np_utils
from keras.layers.pooling import GlobalMaxPooling1D

import Data_IO

model_name = 'ae_cnn_multi_task_NoPretrain_FLOW016_50.h5'
prob = 'FLOW016' # FLOW016, MNMX, SUBINC, SUMTRIAN

model_path = '../ProgramClassification/models_0'
image_name = model_name.split(".")[0] + "_visualize.png"
save_to = "./results/" + image_name
print(save_to)

nb_classes = 5
batch_size = 50

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

file_train = '../asmdata/' + prob +'_Seq_train.txt'
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


model = load_model(os.path.join(model_path, model_name))
latent_map_layer = model.get_layer('latent_map').output
global_pooling = GlobalMaxPooling1D()(latent_map_layer)
latent_map_model = Model(inputs=model.input, outputs=global_pooling)

latent_map = latent_map_model.predict(X_test, batch_size=batch_size)

model.summary()

# print(type(latent_map))
# for map in latent_map:
#     print(type(map))
#     print(map.shape)
#     print(map)

# For speed of computation, only run on a subset
x_data = latent_map
y_data = y_test
y_data = np.argmax(y_data, axis=1)

from sklearn import datasets
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(x_data)
target_ids = [0, 1, 2, 3, 4]
target_names = ["class 0", "class 1", "class 2", "class 3", "class 4"]

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
# colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
colors = 'r', 'g', 'b', 'c', 'm'
for i, c, label in zip(target_ids, colors, target_names):
    plt.scatter(X_2d[y_data == i, 0], X_2d[y_data == i, 1], c=c, label=label)
plt.legend()
plt.savefig(save_to)
print("Saved figure to ", save_to)
