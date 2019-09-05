from scipy.spatial.distance import cosine

def similarity_measure(s1, S1, s2, S2):
  return cosine_similarity(s1, s2)
  # return weighted_cosine_similarity(s1, S1, s2, S2)
  
def cosine_similarity(s1, s2):
  v1 = model.get_vector(s1)
  v2 = model.get_vector(s2)
  
  return 1 - cosine(v1, v2)

def d(s, S):
  return model.vocab[s].count / sum([model.vocab[_s].count for _s in S])

def weighted_cosine_similarity(s1, S1, s2, S2):
  return d(s1, S1) * d(s2, S2) * (cosine_similarity(s1, s2) ** 8)

def get_associated_sense_embeddings(w):
  S = []
  for v in model.vocab:
    t = v.split('_')
    l = ' '.join(t[:-1])
    if w == l and len(t) > 1:
      S.append(v)
      
  return S
