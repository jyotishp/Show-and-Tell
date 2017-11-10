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

# Set Tensorflow backend to avoid full GPU pre-loading
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config = config))

# Read the list of images from img_list_file
def get_image_filenames(filename):
	with open(filename) as f:
		image_filenames = f.read().split('\n')
	# Remove the trailing blank line at the end of the file and return the
	# list of image filenames
	return filter(None, image_filenames)

# Choose a CNN between InceptionV3 and VGG16 and generate image features
def build_image_features(image_filenames, dataset_directory, cnn = 'inception', verbose = True):
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

	preprocessed_images = []
	# Remove repeating filenames (if any)
	# image_filenames = list(set(image_filenames))
	number_of_images = len(image_filenames)
	img_input = []
	
	# Display bar for preprocessing
	if verbose == True:
		print('Preprocessing images:')
		bar = progressbar.ProgressBar(
				term_width = 56,
				max_value = number_of_images,
				widgets = [
					progressbar.Counter(format='%(value)04d/%(max_value)d'),
					progressbar.Bar('=', '[', ']', '.'),
					' ',
					progressbar.ETA()
					])

	# Iterate over all images and preprocess them
	for img_id, img_name in enumerate(image_filenames): # For coco make 3D array , do batch
		img_filepath = dataset_directory + '/Flickr8k_Dataset/' + img_name
		# Image preprocessing
		img = image.load_img(img_filepath, target_size = target_size)
		img = image.img_to_array(img)
		img = np.expand_dims(img, axis=0)
		img = preprocess_input(img)

		# Display the number of files processed
		if img_id % 100 is 0 and verbose == True:
			bar.update(img_id + 100)

		preprocessed_images.append(np.squeeze(img))
	return model, preprocessed_images, output_shape

# Build and save features for all images and save it in cnn_features.h5
# dataset_directory = path to flicker8k dir containing the *.txt files
# img_list_file = File containing list of image filenames
def save_image_features(dataset_directory = '../data/flicker8k', img_list_file = 'Flickr_8k.trainImages.txt', save_name = 'cnn_features.h5'):
	# Get image filenames
	image_filenames = get_image_filenames(dataset_directory + '/' + img_list_file)
	
	# Preprocess the images
	model, preprocessed_images, output_shape = build_image_features(
			image_filenames = image_filenames,
			dataset_directory = dataset_directory)
	print('\nPreprocessing done!\n')

	preprocessed_images = np.asarray(preprocessed_images)
	
	# Build CNN features
	print('Generating features:')
	cnn_features = model.predict(preprocessed_images, batch_size = 128, verbose = 1)
	
	# Save CNN features
	preprocessed_data_directory = dataset_directory + '/preprocessed'
	cnn_features_filepath = preprocessed_data_directory + '/' + save_name
	
	# Create preprocessed folder to save preprocessed files
	if not os.path.exists(preprocessed_data_directory):
		os.mkdir(preprocessed_data_directory)

	# Remove existing features file
	# IDK it throws error if I don't do this :(
	if os.path.exists(cnn_features_filepath):
		os.remove(cnn_features_filepath)

	dataset_file = h5py.File(cnn_features_filepath)
	print('Now saving the CNN features at', cnn_features_filepath)

	for img_id, img_name in enumerate(image_filenames):
		group = dataset_file.create_group(img_name)
		group.create_dataset('cnn_features', output_shape, dtype = 'float32', data = cnn_features[img_id])
	dataset_file.close()
	
if __name__ == "__main__":
	save_image_features(dataset_directory = '/scratch/jyotish/Show-and-Tell/data/flicker8k')