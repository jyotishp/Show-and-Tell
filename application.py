import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.layers import LSTM
from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from generator import Generator
from models import *
import h5py
import numpy as np
import sys
import os
import random
from flask import Flask
from flask import render_template, send_from_directory

app = Flask(__name__)
app.config.from_pyfile('flaskapp.cfg')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config = config))

class Captioner:
    def __init__(self, dataset_directory, 
            batch_size, max_token_len, lstm_model_path,
            cnn_model = 'inception', embedding_size = 512):
        
        self.generator = Generator(dataset_directory = dataset_directory, 
                cnn_model = cnn_model, 
                max_token_len = max_token_len, 
                batch_size = batch_size)
        self.cnn_model = self.getCNNModel(cnn_model)
        self.lstm_model = load_model(lstm_model_path)

    def getCNNModel(self, cnn_model):
        if cnn_model == 'inception':
            # Import InceptionV3 modules
            from keras.applications.inception_v3 import InceptionV3
            from keras.applications.inception_v3 import preprocess_input
          
       	    base_model = InceptionV3(weights = 'imagenet', include_top = True, input_shape = (299, 299, 3))
            model =  Model(
                input = base_model.input,
                outputs = [base_model.get_layer('mixed10').output])
            self.target_size = (299, 299)
            output_shape = model.output_shape[1:]

        elif cnn_model == 'vgg16':
            # Import VGG16 modules
            from keras.applications.vgg16 import VGG16
            from keras.applications.vgg16 import preprocess_input
          
            base_model = VGG16(weights = 'imagenet', include_top = True)
            model = Model(
              	input = base_model.input,
                outputs = [base_model.get_layer('block5_conv3').output])
            self.target_size = (224, 224)
            output_shape = model.output_shape[1:]

        else:
            print('Unknown CNN architecture.')
            sys.exit()
        return model

    def getCaption(self, img_path):
        self.image = image.load_img(img_path, target_size = self.target_size)
        self.image = image.img_to_array(self.image)
        self.processed_image = np.expand_dims(self.image, axis = 0)
        self.processed_image = preprocess_input(self.processed_image)

        cnn_input = np.asarray(self.processed_image)
        cnn_features = self.cnn_model.predict(cnn_input, batch_size = 16, verbose = 0)
        text_in = np.zeros((1, self.generator.max_token_len))
        text_in[0][:] = np.full((self.generator.max_token_len,),0)
        text_in[0][0] = self.generator.token_to_id['<start>']

        predictions = []
        for arg in range(self.generator.max_token_len - 1):
            pred = self.lstm_model.predict([cnn_features, text_in])
            token = np.argmax(pred[0][arg])
            word = self.generator.id_to_token[token]
            text_in[0][arg+1] = token
            if word == '<end>':
                break
            predictions.append(word)
        return predictions, img_path

def randomImage(test_dir_path):
    file_name = random.choice(os.listdir(test_dir_path))
    return test_dir_path + '/' + file_name

captioner = Captioner(
    dataset_directory = 'data', 
    batch_size = 32, 
    max_token_len = 51, 
    lstm_model_path = 'models/attention_50.h5')

@app.route('/')
def hello_word():
    return 'Hi'

@app.route('/img/<path:path>')
def send_js(path):
    return send_from_directory('img', path)

@app.route('/random')
def show_caption_for_random_image():
    caption, filepath = captioner.getCaption(randomImage('./img'))
    return render_template('show_image.html', caption = caption, filepath = filepath)
