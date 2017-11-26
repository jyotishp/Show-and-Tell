import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.layers import LSTM
from keras.models import Sequential, Model
from keras.layers import Dense, Activation,Input, Add, Masking, Embedding, RepeatVector, BatchNormalization, concatenate, Dropout, TimeDistributed, InputLayer
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from generator import Generator
from models import *
import h5py
import numpy as np
import sys
import os

# Set Tensorflow backend to avoid full GPU pre-loading
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.global_variables_initializer()
set_session(tf.Session(config = config))

print "Loading Generator"
gen = Generator(dataset_directory = '/scratch/jyotish/show_and_tell_coco/data')

print "Generating model"
model = captioning_model(
			gen.img_feature_size,
			gen.embedding_size,
			gen.max_token_len,
			gen.vocab_size)
# Load previously saved model if necessary
# from keras.models import load_model

# model.load_weights('/scratch/jyotish/show_and_tell_coco/data/COCO/models/initial_pad_sat_epoch_11.h5')

batch_size = gen.batch_size
print "Start training"
prev_loss = 10

for i in range(1, 10):
	hist = model.fit_generator(
				gen.pullData(),
				epochs=10,
				steps_per_epoch=int(gen.training_samples_count / batch_size), 
				shuffle = True, verbose = 1#, callbacks = [checkpointer]
				)
	if hist.history['loss'][-1] < prev_loss:
		prev_loss = hist.history['loss'][-1]
		print 'Saving ~COCO/models/initial_sat_epoch_' + str(i*10) + '.h5'
		model.save('/scratch/jyotish/show_and_tell_coco/data/COCO/models/attention_' + str(i*10) + '.h5')
