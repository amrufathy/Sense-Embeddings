# -*- coding: utf-8 -*-

import time
import multiprocessing

from utils import *
from parse_datasets import *

bn2wn = get_bn2wn_mapping('bn2wn_mapping.txt')

xml_path = 'EuroSense/eurosense.v1.0.high-precision.xml'
tik = time.time()
eurosense_sents = parse_eurosense(xml_path)
tok = time.time()
print('Parsing eurosense: {} minutes'.format((tok - tik) / 60))

tik = time.time()
sew_sents = parse_sew('sew_conservative/*/*.xml')
tok = time.time()
print('Parsing SEW: {} minutes'.format((tok - tik) / 60))

trainomatic_sents = parse_trainomatic('TRAIN-O-MATIC-DATA/EN/EN.500-2.0/*.xml')


"""## Word2Vec Model"""

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

train_model = False

if train_model:
	train_sents = eurosense_sents + sew_sents + trainomatic_sents

	model = Word2Vec(size=400, window=10, sample=10e-5, 
		 workers=multiprocessing.cpu_count(), hs=1, negative=0, 
		 iter=15, compute_loss=True)
	model.build_vocab(train_sents)

	model.train(train_sents, total_examples=model.corpus_count,
		epochs=model.iter, compute_loss=model.compute_loss)
	model.wv.save_word2vec_format('embeddings.vec', binary=False)
	filter_sense_embeddings('embeddings.vec')

	model = model.wv

else:
	model = KeyedVectors.load_word2vec_format('embeddings.vec', binary=False)


# calculate correlation coefficient
from score import *
r = calculate_score('wordsim353/combined.tab')


# analysis
from visualize import *

# t-SNE plot
all_words = ['seek_01315613v', 'make_up_02520730v',
  'queen_10499355n', 'function_01095218v', 'liner_03673027n']
embeddings_en_2d, word_clusters = get_clusters(all_words)
tsne_plot_similar_words(all_words, embeddings_en_2d, word_clusters,
                          'similar_words.png')


# kNN
w1 = 'bank_09213565n' # river bank 09213565n
w2 = 'bank_08420278n' # financial inst "08420278n"

cw = w2
v1 = model.get_vector(cw)
for sw in model.similar_by_word(cw, topn=10):
  v2 = model.get_vector(sw[0])
  print(cw, sw, jaccard_similarity(v1, v2))


