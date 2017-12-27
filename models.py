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

def get_model_with_attention(cnn_feature_size, vocab_size, max_token_len, embedding_dim = 512):

	#model here
	cnn_X=cnn_feature_size[0]
	cnn_Y=cnn_feature_size[1]
	filters=cnn_feature_size[2]

	emd_word=Sequential()
	emd_word.add(Embedding(vocab_size,embedding_dim,input_length=max_token_len, name='word_embedding'))
	emd_word.add(BatchNormalization())
	# emd_word.summary()

	img_inp=Sequential()
	img_inp.add(Flatten(input_shape=(cnn_X, cnn_Y, filters)))
	img_inp.add(RepeatVector(max_token_len))
	img_inp.add(TimeDistributed(Reshape((cnn_X*cnn_Y,filters))))
	img = TimeDistributed(Flatten())(img_inp.output)
	img = TimeDistributed(Dense(embedding_dim))(img)
	img = TimeDistributed(Dropout(0.3))(img)

	datain = concatenate([emd_word.output,img])   
	att_inp = LSTM(embedding_dim,return_sequences=True,dropout=0.5, name='attention_LSTM')(datain)
	att_inp = TimeDistributed(Dense(cnn_X*cnn_Y, name='attention_Dense'))(att_inp)
	att_inp = TimeDistributed(Activation('softmax' ,name='activation_softmax'))(att_inp)
	att_inp = TimeDistributed((RepeatVector(filters)))(att_inp)
	att_inp = Permute((1,3,2))(att_inp)  

	mult=multiply([img_inp.output,att_inp], name='')
	mult=Permute((1,3,2),input_shape=(max_token_len,filters,cnn_X*cnn_Y))(mult)
	z=TimeDistributed(AveragePooling1D(pool_size=cnn_X*cnn_Y))(mult)
	z=TimeDistributed(BatchNormalization())(z)
	z=TimeDistributed(Reshape((filters,)))(z)
	z=TimeDistributed(Dense(embedding_dim*2))(z)
	z=TimeDistributed(Dropout(0.3))(z)

	lstm_in = concatenate([z, emd_word.output])
	lstm_out = LSTM(1536,return_sequences=True,dropout=0.5)(lstm_in)
	lstm_out = TimeDistributed(Dense(vocab_size,activation='softmax'))(lstm_out)

	model = Model(inputs=[img_inp.input,emd_word.input],outputs=lstm_out)

	return model

def captioning_model(cnn_feature_size,
                    embedding_size,
                    max_token_len,
                    vocab_size):
    cnn_flat_dim = cnn_feature_size[0] * cnn_feature_size[1]
    attention_size = embedding_size
    
    image_embedding_model = Sequential()
    image_embedding_model.add(Reshape((cnn_flat_dim, cnn_feature_size[2]), 
                                       name = 'flatten_cnn',
                                       input_shape = cnn_feature_size))
    image_embedding_model.add(BatchNormalization())
    image_embedding_model.add(GlobalAveragePooling1D(name = 'cnn_avg'))
    image_embedding_model.add(Dense(embedding_size, activation='relu', name='cnn_embedding'))
    image_embedding_model.add(Dropout(0.25))
    image_embedding_model.add(BatchNormalization())
    image_embedding_model.add(RepeatVector(max_token_len))
                              
    word_embedding_model = Sequential()
    word_embedding_model.add(Embedding(vocab_size, embedding_size, name = 'word_embedding', 
                                       input_length = max_token_len))
    word_embedding_model.add(Activation('relu'))
    word_embedding_model.add(BatchNormalization())
    word_embedding_model.add(Dropout(0.25))
                              
    
    attention_image = Dense(attention_size, activation='relu', 
          input_shape = (cnn_flat_dim, cnn_feature_size[2]))(image_embedding_model.get_layer('flatten_cnn').output)
    attention_image = Dropout(0.25)(attention_image)
    attention_image = BatchNormalization()(attention_image)
    attention_image = TimeDistributed(RepeatVector(max_token_len))(attention_image)
    attention_image = Permute((2,1,3))(attention_image)
    
    embedded_attention_image = Dense(embedding_size, activation = 'relu',
                        input_shape = (cnn_flat_dim, 
                                       cnn_feature_size[2]))(image_embedding_model.get_layer('flatten_cnn').output)
    embedded_attention_image = Dropout(0.25, name = 'attention_embedding')(embedded_attention_image)
    embedded_attention_image = BatchNormalization()(embedded_attention_image)
    embedded_attention_image = TimeDistributed(RepeatVector(max_token_len))(embedded_attention_image)
    embedded_attention_image = Permute((2,1,3))(embedded_attention_image)
                              
    lstm_input = concatenate([image_embedding_model.output, word_embedding_model.output])
    attention_features = LSTM(attention_size, return_sequences=True, dropout=0.5,
                               input_shape = (max_token_len, 2*embedding_size))(lstm_input)
    attention_features = Dense(attention_size, activation='tanh', 
                                         name = 'linear_attention_weights')(attention_features)
    attention_features = BatchNormalization()(attention_features)
    linear_attention_weights = attention_features
    attention_features = Dropout(0.25)(attention_features)
    attention_features = Dense(embedding_size, activation = 'relu')(attention_features)
    attention_features = BatchNormalization()(attention_features)
    attention_features = TimeDistributed(RepeatVector(cnn_flat_dim))(attention_features)
    attention_features = add([attention_features, embedded_attention_image])
    attention_features = Dropout(0.5, 
                                 input_shape = (max_token_len, 
                                                cnn_flat_dim, 
                                                embedding_size))(attention_features)
    attention_features = TimeDistributed(Activation('tanh'))(attention_features)
    attention_features = TimeDistributed(Dense(1))(attention_features)
    attention_features = Reshape((max_token_len, cnn_flat_dim))(attention_features)
    attention_features = TimeDistributed(Activation('softmax', name = 'attention'))(attention_features)
    attention_features = TimeDistributed(RepeatVector(attention_size))(attention_features)
    attention_features = Permute((1,3,2))(attention_features)
    
    weighted_cnn_features = TimeDistributed(Lambda(lambda x: K.sum(x, axis=-2), 
                                output_shape = (attention_size,)))(multiply([attention_features, attention_image]))
    weighted_cnn_features = add([weighted_cnn_features, 
                                   linear_attention_weights])
    weighted_cnn_features = BatchNormalization()(weighted_cnn_features)
    
    
    caption = TimeDistributed(Dense(embedding_size, activation='tanh', 
                                           input_shape = (embedding_size, cnn_flat_dim)))(weighted_cnn_features)
    caption = BatchNormalization()(caption)
    caption = TimeDistributed(Dense(vocab_size, activation = 'softmax'), name = 'caption_output')(caption)
    
    model = Model(inputs = [image_embedding_model.input, word_embedding_model.input],
                   outputs = caption)
    return model