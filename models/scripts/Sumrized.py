#!/usr/bin/python3

from helper import Helper, tools
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import gensim.models.keyedvectors as word2vec

help = Helper()

w2vAr_path = tools+'wiki.ar/wiki.ar.vec' #arabe
#w2vAr_path = tools+'wiki.en.vec' #englais


class Sumrized():

    def __init__(self):
        self.word2vec = word2vec.KeyedVectors.load_word2vec_format(w2vAr_path, binary=True,
                                                                   unicode_errors='ignore')
        self.index2word_set = set(self.word2vec.wv.index2word)
        self.word_vectors = dict()
        self.topic_threshold = 0.3
        self.subtract_centroid = False
        self.sim_threshold = 0.95
        count = 0
        self.centroid_space = np.zeros(self.word2vec.vector_size, dtype="float32")
        for w in self.index2word_set:
            self.centroid_space = self.centroid_space + self.word2vec[w]
            count += 1
        if count != 0:
            self.centroid_space = np.divide(self.centroid_space, count)
        return


    def get_bow(self, sentences):
        vectorizer = CountVectorizer()
        sent_word_matrix = vectorizer.fit_transform(sentences)

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
        tfidf = transformer.fit_transform(sent_word_matrix)
        tfidf = tfidf.toarray()

        centroid_vector = tfidf.sum(0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())
        for i in range(centroid_vector.shape[0]):
            if centroid_vector[i] <= self.topic_threshold:
                centroid_vector[i] = 0
        return tfidf, centroid_vector



    def get_topic_idf(self, sentences):
        vectorizer = CountVectorizer()
        sent_word_matrix = vectorizer.fit_transform(sentences)

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
        tfidf = transformer.fit_transform(sent_word_matrix)
        tfidf = tfidf.toarray()

        centroid_vector = tfidf.sum(0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())

        feature_names = vectorizer.get_feature_names()
        word_list = []
        for i in range(centroid_vector.shape[0]):
            if centroid_vector[i] > 0.3:
                word_list.append(feature_names[i])

        return word_list


    def word_vectors_cache(self, sentences):
        self.word_vectors = dict()
        for s in sentences:
            words = s.split()
            for w in words:
                if w in self.index2word_set:
                    if self.subtract_centroid:
                        self.word_vectors[w] = (self.word2vec[w] - self.centroid_space)
                    else:
                        self.word_vectors[w] = self.word2vec[w]
        return


    def compose_vectors(self, words):
        composed_vector = np.zeros(self.word2vec.vector_size, dtype="float32")
        word_vectors_keys = set(self.word_vectors.keys())
        count = 0
        for w in words:
            if w in word_vectors_keys:
                composed_vector = composed_vector + self.word_vectors[w]
                count += 1
        if count != 0:
            composed_vector = np.divide(composed_vector, count)
        return composed_vector


    def summarize(self, text, limit):
        raw_sentences = help.getArticleSentences(text)
        clean_sentences = help.getCleanSentences(raw_sentences)
        centroid_words = self.get_topic_idf(clean_sentences)
        self.word_vectors_cache(clean_sentences)
        centroid_vector = self.compose_vectors(centroid_words)

        sentences_scores = []
        for i in range(len(clean_sentences)):
            words = clean_sentences[i].split()
            sentence_vector = self.compose_vectors(words)
            score = help.similarity(sentence_vector, centroid_vector)
            sentences_scores.append((i, raw_sentences[i], score, sentence_vector))

        sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True)
        count = 0
        sentences_summary = []
        for s in sentence_scores_sort:
            if s[0] == 0:
                sentences_summary.append(s)
                count += 1
                sentence_scores_sort.remove(s)
                break

        for s in sentence_scores_sort:
            if count > limit:
                break
            include = True
            for ps in sentences_summary:
                sim = help.similarity(s[3], ps[3])
                if sim > self.sim_threshold:
                    include = False
            if include:
                sentences_summary.append(s)
                count += 1

        # ordonancement des phrases pour le résumé finale
        sentences_summary = sorted(sentences_summary, key=lambda el: el[0], reverse=False)
        summary = " ".join([s[1] for s in sentences_summary])

        return summary


if __name__ == '__main__':
    sumrized = Sumrized()
    articlePath = '../../tests/4.txt'

    content = help.getArticleContent(articlePath)

    limit = help.getLimit(30, content)
    summary = sumrized.summarize(content, limit)
    print(summary)