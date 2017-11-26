from __future__ import print_function
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.models import Model
from keras.preprocessing import image
import numpy as np
import sys
import os
import h5py
import progressbar
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# Set Tensorflow backend to avoid full GPU pre-loading
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config = config))

# Load keras built in model for inception or VGG16
def get_cnn_model(cnn = 'inception'):
	if cnn == 'inception':
		# Import InceptionV3 modules
		from keras.applications.inception_v3 import InceptionV3
		from keras.applications.inception_v3 import preprocess_input

		base_model = InceptionV3(weights = 'imagenet', include_top = True)
		model =  Model(
				input = base_model.input,
				outputs = [base_model.get_layer('avg_pool').output])
		target_size = (299, 299)
		output_shape = (2048,)

	elif cnn == 'vgg16':
		# Import VGG16 modules
		from keras.applications.vgg16 import VGG16
		from keras.applications.vgg16 import preprocess_input

		base_model = VGG16(weights = 'imagenet', include_top = True)
		model = Model(
				input = base_model.input,
				outputs = [base_model.get_layer('fc2').output])
		target_size = (224, 224)
		output_shape = (4096,)

	else:
		print('Unknown CNN architecture.')
		sys.exit()
	return model, target_size, output_shape, preprocess_input

# Generate image features
def build_image_features(img_ids, coco_instance, 
						 target_size, output_shape,
						 preprocess_input,
						 img_list_file, dataset_directory,
						 progress):
	preprocessed_images = []

	# Iterate over all images and preprocess them
	for pos, img_id in enumerate(img_ids): # For coco make 3D array , do batch
		img_data = coco_instance.loadImgs(img_id)
		img_filepath = dataset_directory + '/'+ img_list_file +'/' + img_data[0]['file_name']
		# Image preprocessing
		img = image.load_img(img_filepath, target_size = target_size)
		img = image.img_to_array(img)
		img = np.expand_dims(img, axis=0)
		img = preprocess_input(img)

		# Display intermediate progress
		if pos % 50 == 0 and len(progress) > 0:
			progress[0].update(progress[1] * progress[2] + pos)
		
		preprocessed_images.append(np.squeeze(img))
	return preprocessed_images

# Build and save features for all images and save it in cnn_features.h5
# dataset_directory = path to extracted MS COCO dir containing the *.json files
# img_list_file = File containing list of training instances
def save_image_features(dataset_directory = '../data/flicker8k', 
						img_list_file = 'Flickr_8k.trainImages.txt', 
						save_name = 'cnn_features.h5',
						images_per_step = 512, batch_size = 64):
	# Get image filenames
	annotations_instance_file = '{}/annotations/instances_{}.json'.format(dataset_directory, img_list_file)
	coco = COCO(annotations_instance_file)
	img_ids = coco.getImgIds()
	
	# Save CNN features
	preprocessed_data_directory = dataset_directory + '/../preprocessed'
	cnn_features_filepath = preprocessed_data_directory + '/' + save_name
	
	# Create preprocessed folder to save preprocessed files
	if not os.path.exists(preprocessed_data_directory):
		os.mkdir(preprocessed_data_directory)

	# Remove existing features file
	if os.path.exists(cnn_features_filepath):
		os.remove(cnn_features_filepath)

	print('Building CNN model')
	model, target_size, output_shape, preprocess_input = get_cnn_model()

	# Display bar for preprocessing
	print('Preprocessing images:')
	bar = progressbar.ProgressBar(
		term_width = 56,
			max_value = len(img_ids),
			widgets = [
				progressbar.Counter(format='%(value)05d/%(max_value)d'),
				' ',
				progressbar.Bar('=', '[', ']', '.'),
				' ',
				progressbar.ETA()
				])
	
	dataset_file = h5py.File(cnn_features_filepath)
	for i in range((len(img_ids) / images_per_step) + 1):
		# Preprocess the images
		if (i+1)*images_per_step < len(img_ids):
			 current_img_ids = img_ids[i*images_per_step : (i+1)*images_per_step]
		else:
			current_img_ids = img_ids[i*images_per_step:]

		preprocessed_images = build_image_features(
				current_img_ids,
				coco,
				target_size,
				output_shape,
				preprocess_input,
				img_list_file,
				dataset_directory,
				[bar, i, images_per_step])
		preprocessed_images = np.asarray(preprocessed_images)
	
		# Build CNN features
		cnn_features = model.predict(preprocessed_images, batch_size = batch_size, verbose = 1)
		for pos, img_id in enumerate(current_img_ids):
			group = dataset_file.create_group(str(img_id))
			group.create_dataset('cnn_features', output_shape, dtype = 'float32', data = cnn_features[pos])
			
	dataset_file.close()

if __name__ == "__main__":
	save_image_features(
		dataset_directory = '/scratch/jyotish/show_and_tell_coco/data/COCO/extracted',
		img_list_file = 'train2014',
		save_name = 'cnn_features.h5',
		images_per_step = 4096,
		batch_size = 128)