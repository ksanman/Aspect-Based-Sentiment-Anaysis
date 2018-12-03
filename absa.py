import nltk
import string
import spacy
import numpy as np
from spacy.symbols import nsubj, NOUN, VERB
from spacy import displacy

with open("opinion-lexicon-English/positive-words.txt") as f:
    postive_words_file = f.read()

with open("opinion-lexicon-English/negative-words.txt") as f:
    negative_words_file = f.read()
    
positive_words =  nltk.word_tokenize(postive_words_file)
negative_words =  nltk.word_tokenize(negative_words_file)

sentance = "Not the best build quality in the world (actually fairly thin plastic) but for $5 what else can you ask for?"

# Classify the sentence as positive or negative:

#find aspects in the sentance.
sentiments = []
words = nltk.word_tokenize(sentance.translate(None, string.punctuation))
for w in words:
    if w in positive_words:
        sentiments.append((w.decode('utf-8'), 1))
    elif w in negative_words:
        sentiments.append((w.decode('utf-8'), -1))
        
print 'Sentiments: ', sentiments

nlp = spacy.load('en')
doc = nlp(sentance.decode('utf-8'))
#displacy.render(doc, style='dep', jupyter=True)
nouns = []
adverbs = []
print 'Dependancy info'
print ''
print ''
for token in doc:
    print 'token: ', token.text
    print 'dependancy: ', token.dep_
    print 'head: ', token.head.text
    print 'head dependancy: ', token.head.dep_
    print 'children: ', [child for child in token.children]
    print ''
    print ''
    if token.pos_ == 'NOUN' and token.dep_ not in ('amod'):
        nouns.append(token)
    if token.dep_ in ('advmod', 'amod'):
        adverbs.append(token)
        

def traverse(node, node_level):
    pair = []
    print "node children: ", [child for child in node.children]
    for child in node.children:
        if child.dep_ in (u'advmod'):
            pair.append(child)
            
        c = traverse(child, node_level)
        
        if not c:
            pass
        else:
            pair.extend(c)

        if child.dep_ in (u'amod', u'neg'):
            pair.append(child)

    if not pair:
        return None
        
    return pair
    
pairs = []
root = [token for token in doc if token.head == token][0]

pairs = traverse(root, 1)
pair_text = [p.text for p in pairs]
print pairs
sent = []
for s in sentiments:
    if s[0] in pair_text:
        sent.append(s[1])
sent_sentiment = max(sent)

neg_pair_dep_count = 0
for p in pairs:
    if p.dep_ == 'neg':
        neg_pair_dep_count += 1

if neg_pair_dep_count > 0 & (neg_pair_dep_count == 1 | neg_pair_dep_count % 2 != 0):
    sent_sentiment = -sent_sentiment

print root
print pairs
print 'Negative' if sent_sentiment < 0 else 'Positive'
