from glob import iglob
from lxml.etree import iterparse, tostring

def parse_eurosense(xml_path):
  sentences = []
  
  context = iterparse(xml_path, events=('start', 'end'))
  for idx, (_, element) in enumerate(context):
    # if valid sentence
    if element.tag == 'sentence' and 'id' in element.attrib:
      # get english text
      eng = element.xpath('text[@lang="en"]')
      
      if not eng or not eng[0].text: continue
      
      sentence = process_text(eng[0].text)

      # get english annotations
      annotations = element.xpath('annotations/annotation[@lang="en"]')
      anchor2lemma, lemma2synset = dict(), dict()

      # extract lemma_synset pairs
      for child in annotations:
        bn = child.text
        anchor = process_text(child.get('anchor').lower())

        if bn in bn2wn and anchor:
          lemma = '_'.join(child.get('lemma').split()).lower()
          anchor2lemma[anchor] = lemma
          lemma2synset[lemma] = bn2wn[bn]

      # replace annotated anchors with lemma_synset pair
      sorted_anchors = sorted(anchor2lemma.keys(), key=len, reverse=True)
      for i, anchor in enumerate(sorted_anchors):
        # check if current anchor was contained in a bigger anchor before
        if anchor not in ' '.join(sorted_anchors[:i]):
          lemm_anchor = anchor2lemma[anchor]
          longest_lemma = get_longest_lemma_from_anchor(lemm_anchor, anchor2lemma.values())
          synset = lemma2synset[longest_lemma]

          old = r'\b{}\b'.format(anchor)
          new = '{}_{}'.format(longest_lemma, synset)
          sentence = re.sub(old, new, sentence)
      
      sentences.append(sentence.lower().split())
      
    element.clear()
  
  return sentences



def parse_sew(path):
  sentences = []

  for i, xml in enumerate(iglob(path)):
    # extract first 4M sentences
    if len(sentences) > 4_000_000:
      print(i)
      break

    context = iterparse(xml, events=('start', 'end'))
    for idx, (_, element) in enumerate(context):

      if element.tag.lower() == 'wikiarticle':
        # article text
        article_text = process_text(element.xpath('text')[0].text)

        # all annotations
        annotations = element.xpath('annotations/annotation')
        for child in annotations:
          bn = child.xpath('babelNetID')
          mention = child.xpath('mention')
          if not bn or not mention or not mention[0].text:
            continue

          bn = bn[0].text
          if bn in bn2wn:
            anchor = process_text(mention[0].text)
            new_anchor = '_'.join(anchor.split())

            # this replacement technique works 90% of the time
            old = r'\b{}\b'.format(anchor)
            new = '{}_{}'.format(new_anchor, bn2wn[bn])
            article_text = re.sub(old, new, article_text, count=1)
            
        # randomly pick 20% of article sentences
        article_sents = article_text.split('\n')
        selected_sents = random.sample(article_sents, int(0.2 * len(article_sents)))
        
        for s in selected_sents:
          sentences.append(s.split())

        # only need one article element
        element.clear()
        break

  return sentences



def parse_trainomatic(path):
  sentences = []

  for i, xml in enumerate(iglob(path)):
    
    context = iterparse(xml, events=('start', 'end'))
    for idx, (_, element) in enumerate(context):
      
      if element.tag.lower() == 'corpus':
        child = element.xpath('lexelt')
        
        if not child: continue

        lemma = child[0].get('item').split('.')[0]
        instances = child[0].xpath('instance')
        
        for ins in instances:
          answer, context = ins.xpath('answer'), ins.xpath('context')
          
          if answer and context:
            wn = ins.xpath('answer/@senseId')[0].split(':')[1]
            pair = '{}_{}'.format(lemma, wn)

            sentence = tostring(context[0], method='text', encoding=str).lower()
            sentence = sentence.replace(lemma, pair)

            sentences.append(sentence.split())

      # only one corpus
      element.clear()
      break
      
  return sentences
