import pickle
import os
import h5py
import time
import numpy as np

class Generator():
    def __init__(self, batch_size=64, max_token_len = 39, img_fea_size = 4096, feature_file = "VGG16_features.h5"):
        self.token_to_id = pickle.load(open('token_to_id.p', 'rb'))
        self.id_to_token = pickle.load(open('id_to_token.p', 'rb'))
        self.id_to_token.update({8829:'<P>'})
        self.token_to_id.update({'<P>':8829})
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.vocab_size = len(self.id_to_token.keys())
        self.feature_size = img_fea_size
        self.feature_dataset = h5py.File(feature_file, 'r')
        f = open("train_captions.txt", 'r')
        lines = f.read().split("\n")
        self.training_samples_count = len(lines)
        f.close()      
    
    def make_empty_batch(self):
        captions_batch = np.zeros((self.batch_size, self.max_token_len ))
        targets_batch = np.zeros((self.batch_size, self.max_token_len, self.vocab_size))
        dummy_img_batch = np.zeros((self.batch_size, 512))
#         images_batch = np.zeros((self.batch_size, self.max_token_len, self.feature_size ))
        images_batch = np.zeros((self.batch_size, self.feature_size ))
        return captions_batch,dummy_img_batch, images_batch, targets_batch
    
    def get_one_hots(self,caption):
        caption_onehot = np.zeros((self.max_token_len, self.vocab_size))
        for cap in caption_onehot:
            cap[8829] = 1
#             cap[8622] = 1
        for pos,token in enumerate(caption):
#             tok_id = self.token_to_id[token]
#             caption_onehot[pos][8622] = 0 
            caption_onehot[pos][8829] = 0 
            caption_onehot[pos][int(token)] = 1
#             print pos,int(token)
#         print np.argmax(caption_onehot, axis=1)    
        target_onehot = np.zeros_like(caption_onehot)
        for cap in target_onehot:
            cap[8829] = 1        
#             cap[8622] = 1        
        target_onehot[:-1,] = caption_onehot[1:, ]
#         print np.argmax(target_onehot, axis=1)
#         print ""
        return caption_onehot, target_onehot

    def get_captions(self,caption):
#         fullcap = np.zeros(self.max_token_len)
        fullcap = np.full(self.max_token_len, 8829)
        target = np.zeros(self.max_token_len)
        for pos, i in enumerate(caption):
            fullcap[pos] = i
        target = np.copy(fullcap)
        target = np.delete(target,0)
#         target = np.append(target,0)
        target = np.append(target,8829)
#         print "Here"
#         print target
#         print caption
#         print np.shape(fullcap)
#         print np.shape(target)
        return fullcap, target

    def get_img_features(self, img_name):
        img_features = self.feature_dataset[img_name]['cnn_features'][:]
        img_input = np.zeros(( self.feature_size ))
        img_input[:] = img_features
        return img_input
    
    def pullData(self):
        f = open("train_captions.txt", 'r')
        bc = 0
        lines = f.read().split("\n")
        captions_batch, dummy_img_batch, images_batch, targets_batch = self.make_empty_batch()
        f.close()
        import random
        while True:
            random.shuffle(lines)
            for line in lines:
                out = line.split('%')
                img_name = out[0]
    #                 print img_name
                caption = out[1:]
                cap_cap, tar_cap = self.get_captions(caption)
                captions_batch[bc, :] = cap_cap
                one_cap, one_targets = self.get_one_hots(caption)
                targets_batch[bc,  :] = one_targets
                try:
                    images_batch[bc, :] = self.get_img_features(img_name)
                except:
                    continue
                bc = bc + 1
                if(bc == self.batch_size):
    #                     yield [{'text_input': captions_batch, 'image_input': images_batch}, {'target_output': targets_batch}]
                    yield [[images_batch, dummy_img_batch, captions_batch], targets_batch]
    #                     time.sleep(0.15)
    #                     yield [images_batch, captions_batch], targets_batch
                    bc = 0
                    captions_batch, dummy_img_batch, images_batch, targets_batch = self.make_empty_batch()

