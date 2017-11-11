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
import h5py
import numpy as np
import sys
import os

# Set Tensorflow backend to avoid full GPU pre-loading
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.global_variables_initializer()
set_session(tf.Session(config = config))

# Ask Alle for details
def get_model(cnn_feature_size, vocab_size, max_token_len, embedding_dim = 512):
	visibletime = 1
	dummy_model=Sequential()
	dummy_model.add(InputLayer(input_shape=(embedding_dim,), name = 'zero'))
	dummy_model.add(RepeatVector(max_token_len - visibletime))

	image_model = Sequential()
	image_model.add(Dense(embedding_dim, input_dim = (2048), activation='elu', name = 'image'))
	image_model.add(BatchNormalization())
	imageout=image_model.output
	imageout=RepeatVector(visibletime)(imageout)

	b=concatenate([imageout,dummy_model.output],axis=1)

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

	intermediate = concatenate([ lang_model.output,b])
	# intermediate = (Dropout(0.1))(intermediate)
	intermediate = LSTM(1024,return_sequences=True,dropout=0.5)(intermediate)
	intermediate = BatchNormalization()(intermediate)
	# intermediate = (Dropout(0.3))(intermediate)
	intermediate = LSTM(1536,return_sequences=True,dropout=0.5)(intermediate)
	intermediate = BatchNormalization()(intermediate)
	intermediate = (Dropout(0.3))(intermediate)
	intermediate = TimeDistributed(Dense(vocab_size,activation='softmax', name='output'))(intermediate)

	model=Model(inputs=[image_model.input,dummy_model.input,lang_model.input],outputs=intermediate)
	# model.summary()
	model.compile('adamax',loss='categorical_crossentropy',metrics=['accuracy'])
	return model

print "Loading Generator"
gen = Generator(dataset_directory = '/scratch/jyotish/show_and_tell_coco/data')

print "Generating model"
model = get_model(gen.img_feature_size, gen.vocab_size, gen.max_token_len, gen.embedding_size)

# Load previously saved model if necessary
from keras.models import load_model

model.load_weights('/scratch/jyotish/show_and_tell_coco/data/COCO/models/initial_1_sat_epoch_1.h5')

batch_size = gen.batch_size
print "Start training"
checkpointer = ModelCheckpoint(filepath='/scratch/jyotish/show_and_tell_coco/data/COCO/models/initial_sat_{epoch:02d}_{val_loss:.2f}', 
							   verbose = 1,
							   monitor = 'loss',
							   period = 1,
							   mode = 'min',
							   save_best_only = True)
model.fit_generator(
				gen.pullData(),
				epochs=10,
				steps_per_epoch=int(gen.training_samples_count / batch_size), 
				shuffle = True, verbose = 1, callbacks = [checkpointer]
			   )

model.save('/scratch/jyotish/show_and_tell_coco/data/COCO/models/initial_1_sat_final.h5')
