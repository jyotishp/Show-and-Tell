import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.layers import LSTM
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import RMSprop
from keras.regularizers import l2
from generator import Generator
import h5py
import numpy as np
import sys
import os
from models import get_model_with_attention 

# Set Tensorflow backend to avoid full GPU pre-loading
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config = config))


if __name__ == '__main__':
		print "Loading Generator"
		# Hardcoding because codes will mostly run from my account.
		# Also adding sys.argv stuff with defaults is a pain.
		gen = Generator(dataset_directory = '/home/siddhartha.l/Show-and-Tell/data')

		print "Generating model (This will take while)"
		model = get_model_with_attention(
					gen.img_feature_size, 
					gen.vocab_size, 
					gen.max_token_len, 
					embedding_dim = gen.embedding_size)
		model.compile('adamax', loss='categorical_crossentropy', metrics=['accuracy'])
		# model.summary()

		batch_size = 64
		print "Start training"
		# Save model for every 10 epochs
		prev_loss = 10
		for i in range(1, 20):
			hist = model.fit_generator(
						gen.pullData(),
						epochs=10,
						steps_per_epoch=int(gen.training_samples_count / batch_size), 
						shuffle = True, verbose = 1#, callbacks = [checkpointer]
						)
			if hist.history['loss'][-1] < prev_loss:
				prev_loss = hist.history['loss'][-1]
				print 'Saving ~models/attention_flickr_epoch_' + str(i*10) + '.h5'
				model.save('/home/siddhartha.l/Show-and-Tell/data/models/attention_flickr_epoch_' + str(i*10) + '.h5')
