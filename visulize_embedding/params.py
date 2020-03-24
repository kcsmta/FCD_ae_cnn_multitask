import os
#
nb_classes = 5 #number of classes
embddingsize = 30 # the length of word vectors
max_len=3074 # max length of a sentence

#convolution w1, w2, ... --> pooling ---> merge ---> dense ---> output
# Convolution
filter_length = 2 # the size of filters
conv_filters = [100, 200] # number of filters for each conv layer
pool_size = 2

num_hid = 200

# activation function
activation ='tanh'
# Training
batch_size = 50
nb_epoch = 50
model_path = "models"

data_part = 0.5 # try 1.0, 0.75, 0.5, 0.25

if not os.path.exists(model_path):
    os.mkdir(model_path)
results_path = "results"
if not os.path.exists(results_path):
    os.mkdir(results_path)

data_path="../asmdata"
