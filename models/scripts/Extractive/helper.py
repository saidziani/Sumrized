#!/usr/bin/python3

import os, pickle, re
from string import punctuation
punctuation += '،؛؟'
import nltk
import numpy as np
from scipy.spatial.distance import cosine


root = '/media/said/DevStuff/PFE/Sumrized/'

tools = root+'Tools/'
farasa = tools+'farasa'
farasaSegmenter = farasa + '/segmenter'

stopWords = open(os.path.join(tools, "arabic-stop-words/list.txt")).read().splitlines()

class Helper():
    def __init__(self, article = False):
        self.article = article

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


    def getLemmaArticle(self, content):
        jarFarasaSegmenter = os.path.join(farasaSegmenter, 'FarasaSegmenterJar.jar')

        tmp = os.path.join(farasaSegmenter, 'tmp')
        if os.path.exists(tmp):
            os.system('rm '+tmp)
        open(tmp, 'w').write(content)

        tmpLemma = os.path.join(farasaSegmenter, 'tmpLemma')
        if os.path.exists(tmpLemma):
            os.system('rm '+tmpLemma)

        os.system('java -jar ' + jarFarasaSegmenter + ' -l true -i ' + tmp + ' -o ' + tmpLemma)

        # os.system('echo "' + content + '" > ' + tmp + ' | java -jar ' + jarFarasaSegmenter + ' -l true -i ' + tmp + ' -o ' + tmpLemma)
        return self.getArticleContent(tmpLemma)


    def getCleanArticle(self, content):
        content = ''.join(c for c in content if c not in punctuation)  
        words = content.split()     
        cleandWords = [w for w in words if w not in stopWords]
        return ' '.join(cleandWords)


    def getBagWordsArticle(self, article):
        content = self.getArticleContent(article)
        cleanArticle = self.getCleanArticle(content)
        lemmaContent = self.getLemmaArticle(cleanArticle)
        cleanArticle = self.getCleanArticle(lemmaContent)
        return cleanArticle.split()


    def pipeline(self, articlePath):
        return ' '.join(self.getBagWordsArticle(articlePath))



    def getCleanSentences(self, sentences):
        return [self.getCleanArticle(sent) for sent in sentences]


    def main(self):
        content = self.getArticleContent(self.article)
        lemma = self.getLemmaArticle(content)
        print(lemma)


    def similarity(self, v1, v2):
        score = 0.0
        if np.count_nonzero(v1) != 0 and np.count_nonzero(v2) != 0:
            score = ((1 - cosine(v1, v2)) + 1) / 2
        return score

    def getLimit(self, limit, content):
        nbSentences = len(self.getArticleSentences(content))
        return ( limit * nbSentences ) / 100