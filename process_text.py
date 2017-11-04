import string
from sets import Set
from collections import Counter
import pickle

class img():
    def __init__(self, name, cap):
        self.name = name
        self.cap = cap

images = []
maxl = 0
vocab = []
    
with open("Flickr8k.token.txt") as textFile:
    lines = [line.split('\t') for line in textFile]
    
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
