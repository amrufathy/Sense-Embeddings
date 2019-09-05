"""## Word Similarity
"""

from scipy.stats import spearmanr
from similarity_utils import *

def calculate_score(path):

	gold_scores, my_scores = [], []

	with open(path) as f:
	  # skip header
	  next(f)
	  
	  for line in f:
	    w1, w2, sim = line.lower().strip().split('\t')
	    
	    S1 = get_associated_sense_embeddings(w1)
	    S2 = get_associated_sense_embeddings(w2)
	    
	    score = -1.0
	    
	    for s1 in S1:
	      for s2 in S2:
		score = max(score, similarity_measure(s1, S1, s2, S2))
	      
	    my_scores.append(score)
	    gold_scores.append(float(sim))
	    
	    
	return spearmanr(gold_scores, my_scores)


