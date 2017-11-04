import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config = config))

from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import sys
import os
import h5py

from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import sys
import os
import h5py

def get_img():
    os.chdir("/home/shanmukh.alle/flickr8k")
    f = open("all_images.txt")
    images = f.read().split('\n')
    f.close()
    return filter(None, images)

def save_img_features(image_files):
    base_model = VGG16(weights='imagenet')
    model =  Model(input=base_model.input, output=base_model.get_layer('fc2').output)
    imgs_features = []
    image_files = list(set(image_files))
    number_of_img = len(image_files)
    img_input = []
    tot = len(image_files)
    for img_id, img_name in enumerate(image_files): # For coco make 3D array , do batch
        img_path = 'Flickr8k_Dataset/'+ img_name
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        if img_id % 100 is 0:
            print str(float(img_id)*100/tot) + " % processed! "
        img_input.append(np.squeeze(x))
        
    img_input = np.asarray(img_input)
    features = model.predict(img_input, batch_size = 128)
    os.system('rm VGG16_features.h5')
    dataset_file = h5py.File('VGG16_features.h5')
    image_files = images_FileNames
    print "Now saving the cnn features"

    for img_id, img_name in enumerate(image_files):
        group = dataset_file.create_group(img_name)
        group.create_dataset('cnn_features', (4096,), dtype = 'float32', data=features[img_id])
    dataset_file.close()
    
if __name__ == "__main__":
    images_FileNames = get_img()
    save_img_features(images_FileNames)    
