#!/usr/bin/python3

import nltk, math, os
from string import punctuation
from stop_words import get_stop_words
import numpy as np
# from theano.scalar import float64
from sklearn.feature_extraction import DictVectorizer
from nltk.tag import pos_tag
from nltk.tag import StanfordPOSTagger

punctuation += '،؟'

root = '/media/said/DevStuff/PFE/Sumrized/'
tools = root+'Tools/'
farasa = tools+'farasa'
farasaSegmenter = farasa + '/segmenter'


class Summary():
    def __init__(self, article=False, lang="en"):
        self.lang = lang    #lang in ["en", "ar"]
        self.article = "data/"+lang+"/"+article


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
            stopWords = open(os.path.join(tools, "arabic-stop-words/list.txt")).read().splitlines()
        else:
            stopWords = get_stop_words('english') 
        tokens = nltk.word_tokenize(sent)
        cleanedTokens = []
        for token in tokens:
            if token not in stopWords:
                word = ''.join(c for c in token if c not in punctuation)
                if word != '':
                    cleanedTokens.append(self.wordToStemme(word))
        return cleanedTokens


    ## Word normalization (stemming)
    def wordToStemme(self, word):
        if self.lang == "ar":
            from nltk.stem.isri import ISRIStemmer
            st = ISRIStemmer()
        else:
            from nltk import SnowballStemmer
            st = SnowballStemmer('english')
        return st.stem(word) 


    ## PoS Tagging
    def getPoSTaggedText(self, text):
        tokenizedText = nltk.word_tokenize(text.lower())
        if self.lang == "ar":
            model = 'utils/stanford-pos-tagger/arabic.tagger'
            jar = 'utils/stanford-pos-tagger/stanford-postagger-3.8.0.jar'
            pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8' )
            taggedText = pos_tagger.tag(tokenizedText)
        else:
            taggedText = pos_tag(tokenizedText)
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
    def getArticleSents(self, param=False):
        text = self.getText()
        sents = self.getParagraghSent(text)
        sentences = []
        for sent in sents:
            if param == True:
                sentences.append(sent)
            else:
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
        sents = self.getArticleSents()
        dic = {}
        for st in sents:
            sents = {}
            pos, sent = st[0], st[1]
            words = self.getSentTokens(sent)
            ratio = self.sentThemRatio(thematicWords, words)
            dic[pos] = ratio
        return dic

    # Feature 2: Sentence position 
    def sentPosFeat(self):
        sents = self.getArticleSents()
        totalNbSents = len(sents)
        dic = {}
        for sent in sents:
            if sent[0] in [ 1, totalNbSents ]:
                ratio = 1
            else:
                min, max = 0.2*totalNbSents, 0.2*2*totalNbSents
                ratio = math.cos((sent[0] - min) * ((1/max) - min))
            dic[sent[0]] = round(ratio, 4)
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
    def getPoSTaggedSents(self):
        sents = self.getArticleSents()
        dic = {}
        for sent in sents:
            posTaggedSent = self.getPoSTaggedText(sent[1])
            dic[sent[0]] = posTaggedSent
        return dic

    def onlyNNP(self, posTaggedSent):
        if self.lang == "ar":
            stopWords = open(os.path.join(tools, "arabic-stop-words/list.txt")).read().splitlines()
        else:
            stopWords = get_stop_words('english') 
        cpt, nouns = 0, []
        for word in posTaggedSent:
            if self.lang == "ar":
                if  word[1].split('/')[0] not in stopWords and word[1].split('/')[1] in ['NN']:
                    cpt +=1
                    nouns.append(word[1].split('/')[0])
            else:
                if  word[0] not in stopWords and word[1] in ['NN']:
                    cpt +=1
                    nouns.append(word[0])
        return cpt, nouns

    def properNounsFeat(self):
        posTaggedSents, dic = self.getPoSTaggedSents(), {}
        for sent in posTaggedSents.items():
            cpt, nouns = self.onlyNNP(sent[1])
            dic[sent[0]] = cpt
        return dic


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


    ##################################################
    def getTF(self, sent):
        words = self.getSentTokens(sent)
        freqDist = nltk.FreqDist(words)
        return freqDist


    def getTFIDF(self, sents):
        sents = [sent[1] for sent in sents]
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(min_df=1)
        vectorizer.fit_transform(sents)
        idf = vectorizer.idf_
        return dict(zip(vectorizer.get_feature_names(), idf))


    def tf_isfFeat(self):
        sents, dic = self.getArticleSents(), {}
        tf_idf = self.getTFIDF(sents)
        for sent in sents: 
            TF, ISF = self.getTF(sent[1]), 0
            words = self.getSentTokens(sent[1])
            for word in words:
                if word in tf_idf.keys():
                    ISF += tf_idf[word] * TF[word]
            if ISF != 0:
                tfisf = round(math.log(ISF) / len(sent), 4)
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

    def load2csv(self):
        import pandas as pd
        data = self.createDataset("columns")
        # features_title = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        features_title = ['thematicRatioFeat', 'sentPosFeat', 'sentLenFeat', 'sentParaPosFeat', 'properNounsFeat', 'sentNumeralsFeat', 'properNounsFeat', 'tf_isfFeat', 'centroidSimFeat']
        dataset = {}
        for (pos, features) in data.items():
            dataset[pos] = features
        dataframe = pd.DataFrame(dataset, columns = features_title)
        csv_name = ''.join(self.article.split('.')[:-1])+ ".csv"
        dataframe.to_csv(csv_name, index=False)


    # Build dataset
    def createDataset(self, method=["lines", "columns"]):
        from time import time
        begin = time(); dicFeat1 = self.thematicRatioFeat();end=time();print('thematicRatioFeat',end-begin)
        begin = time(); dicFeat2 = self.sentPosFeat();end=time(); print('sentPosFeat',end-begin); 
        begin = time(); dicFeat3 = self.sentLenFeat();end=time();print('sentLenFeat',end-begin);
        begin = time(); dicFeat4 = self.sentParaPosFeat();end=time();print('sentParaPosFeat',end-begin);
        begin = time(); dicFeat5 = self.properNounsFeat();end=time();print('properNounsFeat',end-begin);
        begin = time(); dicFeat6 = self.sentNumeralsFeat();end=time();print('sentNumeralsFeat',end-begin);
        begin = time(); dicFeat7 = self.properNounsFeat() #TODO named entities
        begin = time(); dicFeat8 = self.tf_isfFeat();end=time();print('tf_isfFeat',end-begin); 
        begin = time(); dicFeat9 = self.centroidSimFeat();end=time();print('centroidSimFeat',end-begin)
        ft = ['pos', 'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9']

        if method == "lines":
            dic, sentence = [], {}
            # pos_sentence1 = [feat1, feat2, ..., featN]
            # pos_sentence2 = [feat1, feat2, ..., featN]
            #...
            # pos_sentenceN = [feat1, feat2, ..., featN]
            for (f1, f2, f3, f4, f5, f6, f7, f8, f9) in zip(dicFeat1.items(), dicFeat2.items(), dicFeat3.items(), dicFeat4.items(), dicFeat5.items(), dicFeat6.items(), dicFeat7.items(), dicFeat8.items(), dicFeat9.items()):
                sentence = {}
                sentence[ft[1]] = f1[1]; 
                sentence[ft[2]] = f2[1]; 
                sentence[ft[3]] = f3[1]; 
                sentence[ft[4]] = f4[1]; 
                sentence[ft[5]] = f5[1]; 
                sentence[ft[6]] = f6[1]; 
                sentence[ft[7]] = f7[1]; 
                sentence[ft[8]] = f8[1]; 
                sentence[ft[9]] = f9[1];
                dic.append(sentence)

        else:
            dic = {}
            # Feat1 = [val1, val2, ..., valN]
            # Feat2 = [val1, val2, ..., valN]
            #...
            # FeatN = [val1, val2, ..., valN]
            # ft = ['pos', 'thematicRatioFeat', 'sentPosFeat', 'sentLenFeat', 'sentParaPosFeat', 'properNounsFeat', 'sentNumeralsFeat', 'properNounsFeat', 'tf_isfFeat', 'centroidSimFeat']
            feat1 = []; feat2 = []; feat3 = []; feat4 = []; feat5 = []; feat6 = []; feat7 = []; feat8 = []; feat9 = []; 
            for (f1, f2, f3, f4, f5, f6, f7, f8, f9) in zip(dicFeat1.items(), dicFeat2.items(), dicFeat3.items(), dicFeat4.items(), dicFeat5.items(), dicFeat6.items(), dicFeat7.items(), dicFeat8.items(), dicFeat9.items()):
                feat1.append(f1[1]); feat2.append(f2[1]); feat3.append(f3[1]); feat4.append(f4[1]); feat5.append(f5[1]); feat6.append(f6[1]); feat7.append(f7[1]); feat8.append(f8[1]); feat9.append(f9[1])
            dic[ft[1]] = feat1; dic[ft[2]] = feat2; dic[ft[3]] = feat3; dic[ft[4]] = feat4; dic[ft[5]] = feat5; dic[ft[6]] = feat6; dic[ft[7]] = feat7; dic[ft[8]] = feat8; dic[ft[9]] = feat9
        return dic


    def sents2numpy(self):
        data = self.createDataset('lines')
        vectorizer = DictVectorizer(sparse=False)
        data = vectorizer.fit_transform(data)
        data_path = self.article+"_dat" 
        np.save(data_path, data)
        np.save(self.article+"_sents", self.getArticleSents(True))
        return


    def loadFromNumpy(self, type=['data', 'sents']):
        if type == 'data':
            data_path = self.article+"_dat.npy" 
            data = np.load(data_path)
            data = np.array(data, dtype=float64)
            return data
        else:
            sents_path = self.article+"_sents.npy" 
            sents = np.load(sents_path)
            return sents


    def main(self):
        print('Data pre-processing...')
        self.sents2numpy()
        print('Data pre-processing finished.')
        # print(self.loadFromNumpy('sents'))


if __name__ == '__main__':
    summary = Summary()
    articlePath = '/media/said/DevStuff/PFE/Data/CategCorporaAr/data/new/test.txt'
    content = summary.getText(articlePath)
    print(content)
    sents = summary.