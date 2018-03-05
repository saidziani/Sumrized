#!/usr/bin/python3

import nltk, math
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

    ## Build a pre-processed article (with Paragraphs)
    def getPrepArticle(self):
        paragraphs = self.getParagraphs()
        dic = {}
        # cpt = 0
        for pg in paragraphs:
            sents = {}
            sentences = self.getParagraghSent(pg)
            for sent in sentences:
                # cpt += 1
                sents[sentences.index(sent)+1] = sent
            dic[paragraphs.index(pg)] = sents
        return dic

    ## return article sentences
    def getArticleSents(self):
        text = self.getText()
        sents = self.getParagraghSent(text)
        sentences = []
        for sent in sents:
            sentences.append([sents.index(sent)+1, sent])
        return sentences


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

    # Feature 2: Sentence position 
    def sentPosFeat(self):
        sents = self.getArticleSents()
        totalNbSents = len(sents)
        print(totalNbSents)
        dic = {}
        for sent in sents:
            if sent[0] in [ 1, totalNbSents ]:
                ratio = 1
            else:
                min, max = 0.2*totalNbSents, 0.2*2*totalNbSents
                ratio = math.cos((sent[0] - min) * ((1/max) - min))
            dic[sent[0]] = ratio
        return dic

    # Feature 3: Sentence length 
        # [0, if number of words is less than 3, No. of words in the sentence, otherwise]
    def sentLenFeat(self):
        sents = self.getArticleSents()
        dic = {}
        for sent in sents:
            words = self.getSentTokens(sent[1])
            dic[sent[0]] = 0 if len(words) < 3 else len(words) 
        return dic

    # Feature 4: Sentence position in paragraph
        # [1, if it is the first or last sentence of a paragraph, 0, otherwise]
    def sentParaPosFeat(self):
        paragraphs = self.getPrepArticle()
        dic, cpt = {}, 0
        for (index ,pg) in paragraphs.items():
            for (pos, sent) in pg.items():
                cpt += 1
                posPara = 1 if pos in [1, len(pg.items())] else 0
                dic[cpt] = posPara
        return dic

    # Feature 5: Number of proper nouns

    # Feature 6: Sentence numerals 
    def sentNumeralsFeat(self):
        sents = self.getArticleSents()
        dic = {}
        for sent in sents:
            words = self.getSentTokens(sent[1])
            numerals = [num for num in words if num.isnumeric()]
            ratio = round(len(numerals) / len(words), 4)
            dic[sents.index(sent)+1] = ratio
        return dic

    # Feature 7: Number of named entities



    # TF−ISF = log( all words TF ∗ ISF ) / Total words
    # Feature 8: Term Frequency-Inverse Sentence Frequency (TF ISF)
    def getTF(self, sent):
        words = self.getSentTokens(sent)
        freqDist = nltk.FreqDist(words)
        return freqDist

    def getISF(self, word, idSent):
        sents, isf = self.getArticleSents(), 0
        for sent in sents:
            if sent[0] != idSent:
                words = self.getSentTokens(sent[1])
                freqDist = nltk.FreqDist(words)
                isf += freqDist[word] 
        return isf

    def getTFISF(self, sent):
        words, idSent, ISF = self.getSentTokens(sent[1]), sent[0], 0
        # TF_ISF = []
        TF = self.getTF(sent[1])
        for word in words:
            ISF += self.getISF(word, idSent) * TF[word]
            # TF_ISF.append([word, TF[word], self.getISF(word, idSent), idSent])
        return ISF

    def tf_isfFeat(self):
        sents, dic = self.getArticleSents(), {}
        for sent in sents: #TODO
            TFXISF = self.getTFISF(sent)
            if TFXISF != 0 :
                tfisf = round(math.log(TFXISF) / len(sent), 4)
            else:
                tfisf = 0
            dic[sents.index(sent)+1] = tfisf
        return dic


    # Feature 9: Sentence to Centroid similarity
        # Sentence Similarity = cosine sim(sentence, centroid)
    # i.e: centroid is the sentence with the highest tf_isf
    def getCentroid(self):
        tf_isf = self.tf_isfFeat()
        from operator import itemgetter
        sortedTFISF = sorted(tf_isf.items(), key=itemgetter(1), reverse=True)
        centroid = sortedTFISF[0][0]
        return centroid

    # Cos similarity between two sentences
    def cosSim(self, idCentroid, idSent):
        sents = self.getArticleSents()
        sent, centroid = sents[idSent-1], sents[idCentroid-1]
        tfSent, tfCentroid = self.getTF(sent[1]), self.getTF(centroid[1])
        common = set(tfCentroid).intersection(tfSent)
        numerateur = 0
        for word in common:
            numerateur += tfCentroid[word] * tfSent[word]
        powTfCentroid = [pow(val, 2) for val in tfCentroid.values()]
        powTfSent = [pow(val, 2) for val in tfSent.values()]
        centroidSum, sentSum = sum(powTfCentroid), sum(powTfSent)
        denumerateur = math.sqrt(centroidSum * sentSum)
        cos_sim = round(numerateur / denumerateur, 4)
        return cos_sim

    def centroidSimFeat(self):
        sents, idCentroid, dic = self.getArticleSents(), self.getCentroid(), {}
        for sent in sents:
            dic[sents.index(sent)+1] = self.cosSim(idCentroid, sent[0])
        return dic



if __name__ == "__main__":
    article = "article.txt"
    summary = Summary(article, "ar")
    print(summary.centroidSimFeat())
