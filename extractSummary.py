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

    ## read text
    def getText(self):
        return open(self.article, 'r').read()

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
            add = ['أن', 'أو', 'عوض', 'فليس', 'ليس']
            stopWords.extend(add)
            tokens = nltk.word_tokenize(sent)
            cleanedTokens = []
            for token in tokens:
                if token not in stopWords:
                    word = ''.join(c for c in token if c not in punctuation)
                    if word != '':
                        cleanedTokens.append(word)
        return cleanedTokens

    ## Word normalization (stemming)
    def wordToStemme(self, word):
        if self.lang == "ar":
            from nltk.stem.isri import ISRIStemmer
            st = ISRIStemmer()
            stemme = st.stem(word)
        return stemme

    ## PoS Tagging
    def getPoSTaggetText(self, text):
        from nltk.tag import StanfordPOSTagger
        if self.lang == "ar":
            jar = 'stanford-pos-tagger/stanford-postagger-3.8.0.jar'
            model = 'stanford-pos-tagger/arabic.tagger'
            pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8' )
            tokenizedText = nltk.word_tokenize(text.lower())
            taggedText = pos_tagger.tag(tokenizedText)
        return taggedText

    ## Build a pre-processed article
    def getPrepArticle(self):
        paragraphs = self.getParagraphs()
        dic = {}
        for pg in paragraphs:
            sents = {}
            sentences = self.getParagraghSent(pg)
            for sent in sentences:
                sents[sentences.index(sent)] = sent
            dic[paragraphs.index(pg)] = sents
        return dic


    ##~~~~~~  Features for RBM train ~~~~~~##
    def thematicWords(self):
        text = self.getText()
        words = self.getSentTokens(text)
        freqDist = nltk.FreqDist(words)
        from operator import itemgetter
        sortedFreqDist = sorted(freqDist.items(), key=itemgetter(1), reverse=True)
        sortedFreqDistList = [token[0] for token in sortedFreqDist]
        return sortedFreqDistList[:10]


    def sentThemRatio(self, thematicWords, sent):
        sentLength = len(sent)
        common = set(sent).intersection(set(thematicWords))
        noThemWords = len(common)
        return round(noThemWords / sentLength, 4)

    # Feature 1: Thematic Ratio
    def thematicRatioFeat(self):
        thematicWords = self.thematicWords()
        paragraphs = self.getPrepArticle()
        dic = {}
        for (index ,pg) in paragraphs.items():
            sents = {}
            for (pos, sent) in pg.items():
                words = self.getSentTokens(sent)
                ratio = self.sentThemRatio(thematicWords, words)
                sents[pos] = ratio
            dic[index] = sents
        return dic



if __name__ == "__main__":
    article = "article.txt"
    summary = Summary(article, "ar")








    # ##~~~~~~ Old Features ~~~~~~##

    # ## Return the similarity between two sentences
    # def sentSimilarity(self, method, sentA, sentB):
    #     # method: 
    #         # 1 => by set of words, word appear just one time
    #         # 2 => all tokens with repitition 
    #     sentA, sentB = sentA.split(), sentB.split()
    #     if method == 1:
    #         return len(set(sentA).intersection(set(sentB)))
    #     else:
    #         cpt = 0
    #         for word in sentA:
    #             if word in sentB:
    #                 cpt += 1
    #         return cpt

    # ## Check if two sentences are connected (min: one element)
    # def sentsConnected(self, sentA, sentB):
    #     return True if self.sentSimilarity(1, sentA, sentB) > 0 else False


    # #~~~~~~~~~~~~~~~~~~~~~~#    
    # # Summarization Features

    # ## Nb of Similarity with the first sentence:
    # def fstSentFeat(self, method, sent, firstSent):
    #     sent, firstSent = sent.split(), firstSent.split()
    #     return self.sentSimilarity(method, sent, firstSent)


    # ## Length of the sentence:
    # def lenSentFeat(self, sent):
    #     maxLen = 100
    #     return len(sent) / maxLen


    # ## Check if last or first sentence:
    #     # method: 1 => first, 0 => last 
    # def posSentFeat(self, method, sent):
    #     splitedArticle2Sent = self.article.split()
    #     articleLastSent = splitedArticle2Sent[len(splitedArticle2Sent)]
    #     articleFstSent = splitedArticle2Sent[len(splitedArticle2Sent)]
    #     if method == 1:
    #         return True if sent == articleFstSent else False
    #     else:
    #         return True if sent == articleLastSent else False


    # ## Similiraty with title (if possible):
    # def titlesSimFeat(self, sent, title):
    #     return self.sentSimilarity(2, sent, title)  

