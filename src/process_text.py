with open("Flickr8k.token.txt") as textFile:
    lines = [line.split('\t') for line in textFile]
    
class img():
    def __init__(self, name, cap):
        self.name = name
        self.cap = cap

import string
from sets import Set
from collections import Counter
import pickle

images = []
maxl = 0

vocab = []
        
for line in lines:
    fname = line[0].split("#")[0]
    caption = line[1].split("\n")[0].split(".")[0]
    caption = caption.translate(None, string.punctuation).lower()
    caption = caption.split(" ")  
    caption.append("<end>")    
    caption = ["<start>"] + caption
    caption = list(filter(lambda a: a != "", caption))
    vocab.extend(caption)
    maxl = max(maxl, len(caption))
    i = img(fname, caption)
    images.append(i)
    
vocab = Counter(vocab)
keys = vocab.keys()

token_to_id = {key:pos for pos, key in enumerate(keys)}
id_to_token = {pos:key for pos, key in enumerate(keys)}    

pickle.dump(token_to_id, open('token_to_id.p', 'wb') )
pickle.dump(id_to_token, open('id_to_token.p', 'wb') )

f = open("train_captions.txt", 'w')
for line in lines:
    fname = line[0].split("#")[0]
    caption = line[1].split("\n")[0].split(".")[0]
    caption = caption.translate(None, string.punctuation).lower()
    caption = caption.split(" ")  
    caption.append("<end>")    
    caption = ["<start>"] + caption
    caption = list(filter(lambda a: a != "", caption))
    num_caption = map(lambda a: str(token_to_id[a]), caption)
    f.write('%'.join([fname]+num_caption))
    f.write('\n')
f.close()
