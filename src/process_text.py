from __future__ import print_function
import string
from sets import Set
from collections import Counter
import pickle
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import progressbar
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# Variables to set / change
dataset_directory = '/scratch/jyotish/show_and_tell_coco/data/COCO/extracted'
token_file_name = 'train2014'

# initialize COCO api for caption annotations
annFile = '{}/annotations/captions_{}.json'.format(dataset_directory, token_file_name)
coco_caps = COCO(annFile)

# load and display caption annotations
# annIds = coco_caps.getAnnIds(imgIds=img['id']);
# anns = coco_caps.loadAnns(annIds)
# coco_caps.showAnns(anns)

# Load all captions
annotation_ids = coco_caps.getAnnIds();
annotations = coco_caps.loadAnns(annotation_ids)

def progress(annotation_ids):
	bar = progressbar.ProgressBar(
	term_width = 56,
		max_value = len(annotation_ids),
		widgets = [
			progressbar.Counter(),
			' ',
			progressbar.Bar('=', '[', ']', '.'),
			' ',
			progressbar.ETA()
			])
	return bar

max_length = 0
vocab = []

# Build vocabulary
bar = progress(annotation_ids)
print('Building Global Vocabulary:')
for pos, annotation in enumerate(annotations):
	caption = annotation['caption']
	caption = str(caption).translate(None, string.punctuation).lower()
	caption = caption.split(" ")
	caption.append("<end>")
	caption = ["<start>"] + caption
	caption = list(filter(lambda a: a != "", caption))
	vocab.extend(caption)
	max_length = max(max_length, len(caption))
	if pos % 50 == 0:
		bar.update(pos)

# Save map between tokens to IDs and vice versa
vocab = Counter(vocab)
keys = vocab.keys()
print('\n\nMax caption length:', max_length)

# Remove less frequent words
for key in keys:
	if vocab[key] < 1:
		del vocab[key]

# Generate translations for token to id and vice-versa
token_to_id = {key:pos for pos, key in enumerate(keys, 1)}
token_to_id.update({'<p>': 0})
id_to_token = {pos:key for pos, key in enumerate(keys, 1)}    
id_to_token.update({0: '<p>'})

# Save translations
pickle.dump(token_to_id, open(dataset_directory + '/../preprocessed/token_to_id.p', 'wb') )
pickle.dump(id_to_token, open(dataset_directory + '/../preprocessed/id_to_token.p', 'wb') )

# Generate captions that will be used for training and testing
print('\nGenerating tokenized captions:')
with open(dataset_directory + '/../preprocessed/train_captions.txt', 'w') as f:
	bar = progress(annotation_ids)
	for pos, annotation in enumerate(annotations):
		img_id = annotation['image_id']
		caption = annotation['caption']
		caption = str(caption).translate(None, string.punctuation).lower()
		caption = caption.split(" ")
		caption.append("<end>")
		caption = ["<start>"] + caption
		caption = list(filter(lambda a: a != "", caption))
		num_caption = map(lambda a: str(token_to_id[a]), caption)
		f.write('%'.join([str(img_id)] + num_caption))
		f.write('\n')
		if pos % 50 == 0:
			bar.update(pos)
print('\n\nDone!')