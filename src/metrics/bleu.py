import pickle
import nltk

class Caption:
    def __init__(self, name):
        self.name = name
        self.captions = ['','','','','']
        self.result = ''
        
    def add(self, caption_number, caption):
        self.captions[caption_number] = caption
        
test_results = pickle.load(open('../../data/flicker8k/preprocessed/test_results.p', 'rb'))

bleu_score = 0.0
test_set = []
reference_set = []
for img_name in test_results:
    ground_truth = test_results[img_name].captions
    result = test_results[img_name].result
    references = []
    hypothesis = result.split()
    for i in ground_truth:
        references.append(i.split())
    test_set.append(hypothesis)
    reference_set.append(references)

bleu_score = nltk.translate.bleu_score.corpus_bleu(reference_set,test_set)
print ("BLEU4: "+str(bleu_score))
bleu_score = nltk.translate.bleu_score.corpus_bleu(reference_set,test_set,weights = (0.33,0.33,0.33))
print ("BLEU3: "+str(bleu_score))
bleu_score = nltk.translate.bleu_score.corpus_bleu(reference_set,test_set,weights = (0.5,0.5))
print ("BLEU2: "+str(bleu_score))
bleu_score = nltk.translate.bleu_score.corpus_bleu(reference_set,test_set,weights = (1.0,))
print ("BLEU1: "+str(bleu_score))
