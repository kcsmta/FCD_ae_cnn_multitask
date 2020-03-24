

import os



import numpy as np
import sys

import Data_IO

embddingsize = 30 # the length of word vectors


file_wordvec = '../asmdata/vec_embedding_no_ops.txt'
print('Loading data...\n')
print ('Load token-vec: '+ file_wordvec)


max_len=0
embedding_matrix, word2id = Data_IO.loadWordEmbedding(file_wordvec)
probs =["FLOW016","MNMX","SUBINC","SUMTRIAN"]
for prob in probs:
    file_train = '../asmdata/' + prob + '_Seq_train.txt'
    file_CV = '../asmdata/' + prob + '_Seq_CV.txt'
    file_test = '../asmdata/' + prob + '_Seq_test.txt'

    print('\nLoad training data: ' + file_train)
    print('\nLoad CV data: ' + file_CV)
    print('\nLoad test data: ' + file_test)

    p_y_train, p_X_train, maxlen_train = Data_IO.load_ASMSeqData(file_train, word2id)
    p_y_CV, p_X_CV, maxlen_CV = Data_IO.load_ASMSeqData(file_CV, word2id)
    p_y_test, p_X_test, maxlen_test = Data_IO.load_ASMSeqData(file_test, word2id)

    max_len = max(max_len, maxlen_train, maxlen_CV, maxlen_test)

print ("max len = ", max_len)
