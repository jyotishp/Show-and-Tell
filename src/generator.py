import pickle
import os
import h5py
import time
import numpy as np
import random

class Generator():
	def __init__(self, batch_size = 64, max_token_len = 51, 
				cnn_model = 'inception', embedding_size = 512, 
				feature_filename = 'cnn_features.h5',
				dataset_directory = '../data'):
		self.dataset_directory = dataset_directory
		self.flickr_directory = self.dataset_directory + '/COCO'
		self.preprocessed_directory = self.dataset_directory + '/COCO/preprocessed'
		self.token_to_id = pickle.load(open(self.preprocessed_directory + '/token_to_id.p', 'rb'))
		self.id_to_token = pickle.load(open(self.preprocessed_directory + '/id_to_token.p', 'rb'))
		self.batch_size = batch_size
		self.max_token_len = max_token_len
		self.vocab_size = len(self.token_to_id.keys())
		if cnn_model == 'inception':
			self.img_feature_size = (2048)
		elif cnn_model == 'vgg16':
			self.img_feature_size = (14, 14, 512)
		else:
			print 'Unknown CNN architecture'
			sys.exit()
		self.feature_dataset = h5py.File(self.preprocessed_directory + '/' + feature_filename, 'r')
		# This bad. It should not be hard coded!
		self.training_samples_count = 414113
		self.embedding_size = embedding_size
	
	def make_empty_batch(self):
		captions_batch = np.zeros((self.batch_size, self.max_token_len))
		targets_batch = np.zeros((self.batch_size, self.max_token_len, self.vocab_size ))
		images_batch = np.zeros((self.batch_size, self.img_feature_size[0], self.img_feature_size[1], self.img_feature_size[2] ))
		return captions_batch, images_batch, targets_batch
	
	def get_one_hots(self,caption):
		caption_onehot = np.zeros((self.max_token_len, self.vocab_size))
		tokenized_caption = np.zeros(self.max_token_len)
		pos = 0
		for pos,token_id in enumerate(caption):
			tokenized_caption[pos] = int(token_id)
			caption_onehot[pos][int(token_id)] = 1
		for i in range(pos + 1, self.max_token_len):
			caption_onehot[i][0] = 1
		target_onehot = np.zeros_like(caption_onehot)
		target_onehot[:-1,:] = caption_onehot[1:, ] 
		return tokenized_caption, target_onehot

	def get_img_features(self, img_name):
		img_features = self.feature_dataset[img_name]['cnn_features'][:]
		img_input = np.zeros((1, self.img_feature_size[0], self.img_feature_size[1], self.img_feature_size[2]))
		img_input[0, :] = img_features
		return img_input
	
	def pullData(self):
		f = open(self.preprocessed_directory + "/train_captions.txt", 'r')
		bc = 0
		lines = f.read().split("\n")
		captions_batch, images_batch, targets_batch = self.make_empty_batch()
		while True:
			random.shuffle(lines)
			for line in lines:
				out = line.split('%')
				img_name = out[0]
				caption = out[1:]
				captions_batch[bc, :], targets_batch[bc, :, :] = self.get_one_hots(caption)
				try:
					images_batch[bc, :] = self.get_img_features(img_name)
				except Exception as e:
					continue
				bc = bc + 1
				if(bc == self.batch_size):
					zero_input = np.full((self.batch_size, self.img_feature_size[0] * self.img_feature_size[1]), 1)
					# IDK why it doesn't work :(
# 					yield [{'text_input': captions_batch, 'image_input': images_batch, 'zero': zero_input}, {'output': targets_batch}]
					yield [[images_batch, captions_batch, zero_input], [targets_batch]] 
					bc = 0
					captions_batch, images_batch, targets_batch = self.make_empty_batch()
		f.close()
