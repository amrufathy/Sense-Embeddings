"""# Visualize Embeddings"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

"""### PCA"""

w1, w2 = 'love', 'hate'
n1, n2 = get_associated_sense_embeddings(w1), get_associated_sense_embeddings(w2)
all_words = n1
all_words.extend(n2)


def visualize_pca(words):
    V = model[words]
    pca = PCA(n_components=2)
    result = pca.fit_transform(V)
    plt.scatter(result[:, 0], result[:, 1])

    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    plt.grid(True)
    plt.show()

"""### t-SNE"""

# Source: https://towardsdatascience.com/google-news-and-leo-tolstoy-visualizing-word2vec-word-embeddings-with-t-sne-11558d8bd4d
def get_clusters(all_words):
  embedding_clusters = []
  word_clusters = []
  for word in all_words:
    embeddings = []
    words = []
    for similar_word, _ in model.most_similar(word, topn=15):
        words.append(similar_word)
        embeddings.append(model[similar_word])
    embedding_clusters.append(embeddings)
    word_clusters.append(words)

  embedding_clusters = np.array(embedding_clusters)
  n, m, k = embedding_clusters.shape
  tsne_model_en_2d = TSNE(
      perplexity=15, n_components=2, init='pca',
      n_iter=3500, random_state=0)
  embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(
      embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
  return embeddings_en_2d, word_clusters

def tsne_plot_similar_words(labels, embedding_clusters, word_clusters, filename=None):
  plt.figure(figsize=(16, 9))
  colors = cm.rainbow(np.linspace(0, 1, len(labels)))
  for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
    x = embeddings[:, 0]
    y = embeddings[:, 1]
    plt.scatter(x, y, c=color, alpha=1.0, label=label)
    for i, word in enumerate(words):
      plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                       textcoords='offset points', ha='right', va='bottom', size=8)
  plt.legend(loc=4)
  plt.grid(True)
  if filename:
      plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
  plt.show()



"""### kNN"""

def jaccard_similarity(v1, v2):
    intersection = np.dot(v1, v2)
    union = (np.linalg.norm(v1) * 2 +
                   np.linalg.norm(v2) * 2 - intersection)
    return np.round(intersection / union, 3)
