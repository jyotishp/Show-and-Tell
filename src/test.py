# model.summary()
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.layers import LSTM
from keras.models import Sequential, Model
from keras.layers import Dense, Activation,Input, Add, Masking, Embedding, RepeatVector, BatchNormalization, concatenate, Dropout, TimeDistributed, InputLayer
from keras.optimizers import RMSprop
from keras.regularizers import l2
from generator import Generator
import h5py
import numpy as np
import sys
import os
import pickle

# Set Tensorflow backend to avoid full GPU pre-loading
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config = config))

print "Loading Gen"
print "Generating model"
gen = Generator()

# returns a compiled model
# identical to the previous one
from keras.models import load_model
model = load_model('../data/notebook_final_model_epoch_90.h5')

print "Loading test captions"
with open('../data/flicker8k/preprocessed/test_captions.txt') as captions_file:
    captions = captions_file.read().split('\n')
    
class Caption:
    def __init__(self, name):
        self.name = name
        self.captions = ['','','','','']
        self.result = ''
        
    def add(self, caption_number, caption):
        self.captions[caption_number] = caption
        
test_results = {}

print "Generating results"
for caption in captions:
    caption = caption.split('\t')
    img_name = caption[0].split('#')
    caption_number = int(img_name[1])
    img_name = img_name[0]
    caption = caption[1].lower()

    feature_dataset = h5py.File('../data/flicker8k/preprocessed/test_features.h5', 'r')
    img_features = feature_dataset[img_name]['cnn_features'][:]

    # image_filenames = get_image_filenames(dataset_directory + '/' + img_list_file)

    # print img_features.shape
    features = np.array([img_features])

    text_in = np.zeros((1,gen.max_token_len))
    text_in[0][:] = np.full((gen.max_token_len,), 0)
    text_in[0][0] = 4230

    # print features,text_in
    arr = []
    zeros = np.array([np.zeros(512)])
    for arg in range(gen.max_token_len-1):
        pred = model.predict([features, zeros, text_in])
        tok = np.argmax(pred[0][arg])
        word = gen.id_to_token[tok]
        text_in[0][arg+1] = tok
        if word == '<end>':
            break
        arr.append(word)
    
    arr.append('.')
    
    try:
        cap_obj = test_results[img_name]
        cap_obj.add(caption_number, caption)
    except Exception as e:
	print "Created:", img_name
        cap_obj = Caption(img_name)
        cap_obj.add(caption_number, caption)
        cap_obj.result = ' '.join(arr)
        test_results.update({img_name: cap_obj})

print len(test_results)
pickle.dump(test_results, open('../data/flicker8k/preprocessed/test_results.p', 'wb') )
