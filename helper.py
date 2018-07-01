#!/usr/bin/python3

import os, pickle, re
from string import punctuation
punctuation += '،؛؟'
import nltk
import numpy as np
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords


tools = 'tools/'

class Helper():
    def __init__(self, lang=['en', 'ar']):
        self.lang = lang
        self.stopWords = set(stopwords.words('english')) if lang=='en' else open(os.path.join(tools, "arabic-stop-words/list.txt")).read().splitlines() 

    ##~~Pickle helpers~~#
    def getPickleContent(self, pklFile):
        with open (pklFile, 'rb') as fp:
            itemlist = pickle.load(fp)
        return itemlist

    def setPickleContent(self, fileName, itemList):
        with open(fileName+'.pkl', 'wb') as fp:
            pickle.dump(itemList, fp)
    #~~~~~~~~~~~~~~~~~~#

    def getArticleContent(self, article):
        if os.path.exists(article):
            return open(article, 'r').read()

    def getArticleSentences(self, content):
        return [sent for sent in nltk.sent_tokenize(content) if len(sent) > 10]

    def getCleanArticle(self, content):
        content = ''.join(c for c in content if c not in punctuation)  
        words = content.split()     
        cleandWords = [w for w in words if w not in self.stopWords]
        return ' '.join(cleandWords)

    def getCleanSentences(self, sentences):
        return [self.getCleanArticle(sent) for sent in sentences]

    def similarity(self, v1, v2):
        score = 0.0
        if np.count_nonzero(v1) != 0 and np.count_nonzero(v2) != 0:
            score = ((1 - cosine(v1, v2)) + 1) / 2
        return score

    def getLimit(self, limit, nbSentences):
        return ( limit * nbSentences ) / 100