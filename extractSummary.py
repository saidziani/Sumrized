#!/usr/bin/python3

import nltk
from string import punctuation
from stop_words import get_stop_words

punctuation += '،؟'


class Summary():
    def __init__(self, article = False, lang = "en"):
        self.article = article
        self.lang = lang    #lang in ["en", "ar"]


    # Useful methods (pre-processing)

    ## Split text to paragraphs
    def getParagraphs(self):
        paragraphs = open(self.article, 'r').readlines()
        return paragraphs
    
    ## Split paragraph to sents
    def getParagraghSent(self, paragraph):
        sents = nltk.sent_tokenize(paragraph)
        return sents

    ## Split sent to tokens
    def getSentTokens(self, sent):
        if self.lang == "ar":
            stopWords = get_stop_words('arabic') 
            tokens = nltk.word_tokenize(sent)
            cleanedTokens = []
            for token in tokens:
                if token not in stopWords:
                    cleanedTokens.append(''.join(c for c in token if c not in punctuation))
        return cleanedTokens

    ## Word normalization
    def wordToStemme(self, word):
        if self.lang == "ar":
            from nltk.stem.isri import ISRIStemmer
            st = ISRIStemmer()
            stemme = st.stem(word)
        return stemme

    def getPoSTaggetText(self, text):
        from nltk.tag import StanfordPOSTagger
        if self.lang == "ar":
            jar = 'stanford-pos-tagger/stanford-postagger-3.8.0.jar'
            model = 'stanford-pos-tagger/arabic.tagger'
            pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8' )
            tokenizedText = nltk.word_tokenize(text.lower())
            taggedText = pos_tagger.tag(tokenizedText)
        return taggedText

    ##~~~~~~ Features ~~~~~~##

    ## Return the similarity between two sentences
    def sentSimilarity(self, method, sentA, sentB):
        # method: 
            # 1 => by set of words, word appear just one time
            # 2 => all tokens with repitition 
        sentA, sentB = sentA.split(), sentB.split()
        if method == 1:
            return len(set(sentA).intersection(set(sentB)))
        else:
            cpt = 0
            for word in sentA:
                if word in sentB:
                    cpt += 1
            return cpt

    ## Check if two sentences are connected (min: one element)
    def sentsConnected(self, sentA, sentB):
        return True if self.sentSimilarity(1, sentA, sentB) > 0 else False


    #~~~~~~~~~~~~~~~~~~~~~~#    
    # Summarization Features

    ## Nb of Similarity with the first sentence:
    def fstSentFeat(self, method, sent, firstSent):
        sent, firstSent = sent.split(), firstSent.split()
        return self.sentSimilarity(method, sent, firstSent)


    ## Length of the sentence:
    def lenSentFeat(self, sent):
        maxLen = 100
        return len(sent) / maxLen


    ## Check if last or first sentence:
        # method: 1 => first, 0 => last 
    def posSentFeat(self, method, sent):
        splitedArticle2Sent = self.article.split()
        articleLastSent = splitedArticle2Sent[len(splitedArticle2Sent)]
        articleFstSent = splitedArticle2Sent[len(splitedArticle2Sent)]
        if method == 1:
            return True if sent == articleFstSent else False
        else:
            return True if sent == articleLastSent else False


    ## Similiraty with title (if possible):
    def titlesSimFeat(self, sent, title):
        return self.sentSimilarity(2, sent, title)  


if __name__ == "__main__":
    article = "article.txt"
    summary = Summary(article, "ar")
    paragraphs = summary.getParagraphs()
    sents = summary.getParagraghSent(paragraphs[0])
    words = summary.getSentTokens(sents[7])
    posTaggedText = summary.getPoSTaggetText(paragraphs[0])
    print(posTaggedText)