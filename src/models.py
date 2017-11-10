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

# Model for Show and Tell without attention
def get_model_no_atenttion(cnn_feature_size, vocab_size, max_token_len, embedding_dim = 512):
	#number of timesteps encoded image vector is shown to the LSTM decoder 
    visibletime = 1
    
    #used to pad the encoded image with zeros for timesteps image vector is not shown to the LSTM decoder
	dummy_model=Sequential()
	dummy_model.add(InputLayer(input_shape=(embedding_dim,), name = 'zero'))
	dummy_model.add(RepeatVector(max_token_len - visibletime))

	#generates image embedding 
    image_model = Sequential()
	image_model.add(Dense(embedding_dim, input_dim = (2048), activation='elu', name = 'image'))
	image_model.add(BatchNormalization())
	imageout=image_model.output
	imageout=RepeatVector(visibletime)(imageout)

    #concatenate the image embedding vector eith the zero padding
	b=concatenate([imageout,dummy_model.output],axis=1)

    #generates the language embedding 
	lang_model = Sequential()
	lang_model.add(Embedding(vocab_size, 256, input_length=max_token_len, name = 'text'))
	lang_model.add(BatchNormalization())
	lang_model.add(LSTM(256,return_sequences=True))
	lang_model.add(BatchNormalization())
	lang_model.add(Dropout(0.3))
	lang_model.add(TimeDistributed(Dense(embedding_dim)))
	lang_model.add(BatchNormalization())

	#concatenate the embedded data of words and images
    intermediate = concatenate([ lang_model.output,b])

    #Two Layer decoder LSTM
	intermediate = LSTM(1024,return_sequences=True,dropout=0.5)(intermediate)
	intermediate = BatchNormalization()(intermediate)
	intermediate = LSTM(1536,return_sequences=True,dropout=0.5)(intermediate)
	intermediate = BatchNormalization()(intermediate)
	intermediate = (Dropout(0.3))(intermediate)
	intermediate = TimeDistributed(Dense(vocab_size,activation='softmax', name='output'))(intermediate)

	#Generates a model combining the imagemodel, languagemodel, decoder 
    model=Model(inputs=[image_model.input,dummy_model.input,lang_model.input],outputs=intermediate)

	# model.summary()
	return model

def get_model_with_atenttion(cnn_feature_size, vocab_size, max_token_len, embedding_dim = 512):
	#model here
	return