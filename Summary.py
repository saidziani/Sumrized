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
            add = ['أن', 'أو', 'عوض', 'فليس', 'ليس', 'حين', 'مع']
            stopWords.extend(add)
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
            stemme = st.stem(word)
        return stemme

    ## PoS Tagging
    def getPoSTaggedText(self, text):
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
        stopWords = get_stop_words('arabic') 
        add = ['أن', 'أو', 'عوض', 'فليس', 'ليس', 'حين', 'مع', 'رغم', 'معه', 'وله', 'له']
        stopWords.extend(add)
        cpt, nouns = 0, []
        for word in posTaggedSent:
            if  word[1].split('/')[0] not in stopWords and  word[1].split('/')[1] in ['NN']:
                cpt +=1
                nouns.append(word[1].split('/')[0])
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

    def load2csv(self, csv_path):
        import pandas as pd
        data = self.createDataset("columns")
        features_title = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # features_title = ['thematicRatioFeat', 'sentPosFeat', 'sentLenFeat', 'sentParaPosFeat', 'properNounsFeat', 'sentNumeralsFeat', 'properNounsFeat', 'tf_isfFeat', 'centroidSimFeat']
        dataset = {}
        for (pos, features) in data.items():
            dataset[pos] = features
        print(dataset)
        dataframe = pd.DataFrame(dataset, columns = features_title)
        dataframe.to_csv(csv_path, index=False)


    # Build dataset
    def createDataset(self, method=["lines", "columns"]):
        dicFeat1 = self.thematicRatioFeat()
        dicFeat2 = self.sentPosFeat()
        dicFeat3 = self.sentLenFeat()
        dicFeat4 = self.sentParaPosFeat()
        dicFeat5 = self.properNounsFeat()
        dicFeat6 = self.sentNumeralsFeat()
        dicFeat7 = self.properNounsFeat()
        dicFeat8 = self.tf_isfFeat()
        dicFeat9 = self.centroidSimFeat()
        dic = {}
        if method == "lines":
            # pos_sentence1 = [feat1, feat2, ..., featN]
            # pos_sentence2 = [feat1, feat2, ..., featN]
            #...
            # pos_sentenceN = [feat1, feat2, ..., featN]
            for (f1, f2, f3, f4, f5, f6, f7, f8, f9) in zip(dicFeat1.items(), dicFeat2.items(), dicFeat3.items(), dicFeat4.items(), dicFeat5.items(), dicFeat6.items(), dicFeat7.items(), dicFeat8.items(), dicFeat9.items()):
                l = [f1[1], f2[1], f3[1], f4[1], f5[1], f6[1], f7[1], f8[1], f9[1]]
                dic[f1[0]] = l
        else:
            # Feat1 = [val1, val2, ..., valN]
            # Feat2 = [val1, val2, ..., valN]
            #...
            # FeatN = [val1, val2, ..., valN]
            feat1 = []; feat2 = []; feat3 = []; feat4 = []; feat5 = []; feat6 = []; feat7 = []; feat8 = []; feat9 = []; 
            for (f1, f2, f3, f4, f5, f6, f7, f8, f9) in zip(dicFeat1.items(), dicFeat2.items(), dicFeat3.items(), dicFeat4.items(), dicFeat5.items(), dicFeat6.items(), dicFeat7.items(), dicFeat8.items(), dicFeat9.items()):
                feat1.append(f1[1]); feat2.append(f2[1]); feat3.append(f3[1]); feat4.append(f4[1]); feat5.append(f5[1]); feat6.append(f6[1]); feat7.append(f7[1]); feat8.append(f8[1]); feat9.append(f9[1])
            dic[1] = feat1; dic[2] = feat2; dic[3] = feat3; dic[4] = feat4; dic[5] = feat5; dic[6] = feat6; dic[7] = feat7; dic[8] = feat8; dic[9] = feat9
        return dic

    def main(self):
        self.load2csv('train.csv')


if __name__ == "__main__":
    article = "article.txt"
    summary = Summary(article, "ar")
    summary.main()

