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

# Set Tensorflow backend to avoid full GPU pre-loading
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config = config))

# Shanmukh Alle is a better person to comment here :)
def get_model(cnn_feature_size, vocab_size, max_token_len, embedding_dim = 512):
	visibletime = 1
	dummy_model=Sequential()
	dummy_model.add(InputLayer(input_shape=(embedding_dim,), name = 'zero'))
	dummy_model.add(RepeatVector(max_token_len - visibletime))
	# print 'lol'

	image_model = Sequential()
	image_model.add(Dense(embedding_dim, input_dim = (2048), activation='elu', name = 'image'))
	image_model.add(BatchNormalization())
	imageout=image_model.output
	imageout=RepeatVector(visibletime)(imageout)
	# print 'lol1'

	b=concatenate([imageout,dummy_model.output],axis=1)
	# print 'lol2'

	# b = Lambda(lambda x: backend.concatenate([x,mask],axis=1),output_shape=(max_token_len,cnn_feature_size))(imageout)
	# image_model.add(RepeatVector(max_token_len))

	lang_model = Sequential()
# 	lang_model.add(InputLayer(input_shape=(max_token_len), name = 'text'))
# 	lang_model.add(Masking(mask_value=0))
	lang_model.add(Embedding(vocab_size, 256, input_length=max_token_len, name = 'text'))
	lang_model.add(BatchNormalization())
	# lang_model.add(TimeDistributed(Dense(128)))
	lang_model.add(LSTM(256,return_sequences=True))
	lang_model.add(BatchNormalization())
	lang_model.add(Dropout(0.3))
	lang_model.add(TimeDistributed(Dense(embedding_dim)))
	lang_model.add(BatchNormalization())
	# print 'lol3'

	intermediate = concatenate([ lang_model.output,b])
	# intermediate = (Dropout(0.1))(intermediate)
	intermediate = LSTM(1024,return_sequences=True,dropout=0.5)(intermediate)
	intermediate = BatchNormalization()(intermediate)
	# intermediate = (Dropout(0.3))(intermediate)
	intermediate = LSTM(1536,return_sequences=True,dropout=0.5)(intermediate)
	intermediate = BatchNormalization()(intermediate)
	intermediate = (Dropout(0.3))(intermediate)
	intermediate = TimeDistributed(Dense(vocab_size,activation='softmax', name='output'))(intermediate)
	# print 'lol4'

	model=Model(inputs=[image_model.input,dummy_model.input,lang_model.input],outputs=intermediate)
	# print 'lol5'
	# model.summary()
	model.compile('rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
	# print 'lol6'
	return model

if __name__ == '__main__':
	print "Loading Generator"
	# Hardcoding because codes will mostly run from my account.
	# Also adding sys.argv stuff with defaults is a pain.
	gen = Generator(dataset_directory = '/scratch/jyotish/Show-and-Tell/data')

	print "Generating model (This will take while)"
	model = get_model(gen.img_feature_size, gen.vocab_size, gen.max_token_len, gen.embedding_size)
	# model.summary()

	batch_size = 64
	print "Start training"
	# Save model for every 10 epochs
	for i in range(15):
		model.fit_generator(gen.pullData(), epochs=10, steps_per_epoch=int(gen.training_samples_count / batch_size), verbose = 1)
		model.save(gen.preprocessed_directory + '/final_model_epoch_' + str(i*10) + '.h5')