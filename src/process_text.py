import string
from sets import Set
from collections import Counter
import pickle

# Variables to set / change
dataset_directory = '/scratch/jyotish/Show-and-Tell/data/flicker8k'
token_file_name = 'Flickr8k.token.txt'

# Read tokens file
with open(dataset_directory + '/' + token_file_name) as textFile:
	lines = [line.split('\t') for line in textFile]

max_length = 0
vocab = []

# Build vocabulary
for line in lines:
	filename = line[0].split("#")[0]
	caption = line[1].split("\n")[0].split(".")[0]
	caption = caption.translate(None, string.punctuation).lower()
	caption = caption.split(" ")  
	caption.append("<end>")    
	caption = ["<start>"] + caption
	caption = list(filter(lambda a: a != "", caption))
	vocab.extend(caption)
	max_length = max(max_length, len(caption))

# Save map between tokens to IDs and vice versa
vocab = Counter(vocab)
keys = vocab.keys()

# Remove less frequent words
for key in keys:
	if vocab[key] < 1:
		del vocab[key]

token_to_id = {key:pos for pos, key in enumerate(keys, 1)}
token_to_id.update({'<p>': 0})
id_to_token = {pos:key for pos, key in enumerate(keys, 1)}    
id_to_token.update({0: '<p>'})

pickle.dump(token_to_id, open(dataset_directory + '/preprocessed/token_to_id.p', 'wb') )
pickle.dump(id_to_token, open(dataset_directory + '/preprocessed/id_to_token.p', 'wb') )

# Generate captions that will be used for training and testing
with open(dataset_directory + '/preprocessed/train_captions.txt', 'w') as f:
	for line in lines:
		filename = line[0].split("#")[0]
		caption = line[1].split("\n")[0].split(".")[0]
		caption = caption.translate(None, string.punctuation).lower()
		caption = caption.split(" ")  
		caption.append("<end>")    
		caption = ["<start>"] + caption
		caption = list(filter(lambda a: a != "", caption))
		num_caption = map(lambda a: str(token_to_id[a]), caption)
		f.write('%'.join([filename] + num_caption))
		f.write('\n')
