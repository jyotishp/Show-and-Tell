import pickle
import os
import h5py
import time
import numpy as np

class Generator():
    def __init__(self, batch_size=64, max_token_len = 39, img_fea_size = 4096, feature_file = "VGG16_features.h5"):
        self.token_to_id = pickle.load(open('token_to_id.p', 'rb'))
        self.id_to_token = pickle.load(open('id_to_token.p', 'rb'))
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.vocab_size = len(self.token_to_id.keys())
        self.feature_size = img_fea_size
        self.feature_dataset = h5py.File(feature_file, 'r')
    
    def make_empty_batch(self):
        captions_batch = np.zeros((self.batch_size, self.max_token_len, self.vocab_size ))
        targets_batch = np.zeros((self.batch_size, self.max_token_len, self.vocab_size ))
        images_batch = np.zeros((self.batch_size, self.max_token_len, self.feature_size ))
        return captions_batch, images_batch, targets_batch
    
    def get_one_hots(self,caption):
        caption_onehot = np.zeros((self.max_token_len, self.vocab_size))
        for pos,token in enumerate(caption):
#             tok_id = self.token_to_id[token]
#             print token
            caption_onehot[pos][int(token)] = 1
        target_onehot = np.zeros_like(caption_onehot)
        target_onehot[:-1,:] = caption_onehot[1:, ]
        return caption_onehot, target_onehot

    def get_img_features(self, img_name):
        img_features = self.feature_dataset[img_name]['cnn_features'][:]
        img_input = np.zeros((self.max_token_len, self.feature_size ))
        img_input[0, :] = img_features
        return img_input
    
    def pullData(self):
#         os.system('sort -R train_captions.txt > shuffled_train_captions.txt')
#         f = open("shuffled_train_captions.txt", 'r')
        f = open("train_captions.txt", 'r')
        bc = 0
        lines = f.read().split("\n")
        captions_batch, images_batch, targets_batch = self.make_empty_batch()
        while True:
            for line in lines[:10]:
                out = line.split('%')
                img_name = out[0]
                caption = out[1:]
                captions_batch[bc, :, :], targets_batch[bc,  :, :] = self.get_one_hots(caption)
                images_batch[bc,  :, :] = self.get_img_features(img_name)
                bc = bc + 1
                if(bc == self.batch_size):
                        yield {'text': captions_batch, 'image': images_batch, 'output': targets_batch}
                    bc = 0
                    captions_batch, images_batch, targets_batch = self.make_empty_batch()
        f.close()
