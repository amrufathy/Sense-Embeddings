import string
import re

def get_bn2wn_mapping(path):
  """
  Returns a dictionary with a mapping between
    BabelNet synsets and WordNet synsets
  """
  bn2wn = dict()

  with open(path) as f:
    for line in f:
      # TODO: check the line with 3 entries
      bn, wn = line.strip().split()[:2]
      bn2wn[bn] = wn
  
  return bn2wn


def process_text(s):
  """
  Removes punctuation and multiple consecutive
    spaces from text
  """
  # remove punctuation characters
  s = s.translate(
     str.maketrans('', '', string.punctuation))
  # remove multiple consecutive spaces
  s = re.sub(' +', ' ', s)
  
  return s.lower()


def get_longest_lemma_from_anchor(lemm_anchor, lemmas):
  """
  Returns the longest lemma containing the `anchor`
    string. According to high precision specification of Eurosense.
  """
  relevant_lemmas = list(filter(lambda x: lemm_anchor in x, lemmas))
  longest_lemma = max(relevant_lemmas, key=len)
  
  return longest_lemma


def filter_sense_embeddings(path):
  """
  Removes word embeddings from a word2vec
    formatted embeddings file
  """
  senses = []
  with open(path, 'r') as f:
      for line in f:
          key = line.split(' ', 1)[0]
          if '_' in key:
              senses.append(line)

  with open(path, 'w') as f:
      f.write("{} {}\n".format(len(senses), len(senses[0].split(' ')) - 1))
      for sense in senses:
          file.write(sense + '\n')
