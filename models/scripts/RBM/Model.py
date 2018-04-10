#!/usr/bin/python3

import numpy as np
from theano.scalar import float64
import rbm
from operator import itemgetter
import os

def rbm_model(article_name, lang=['en', 'ar']):
    data_path= "data/"+lang+"/"+article_name+"_dat.npy"
    sents_path="data/"+lang+"/"+article_name+"_sents.npy"

    data = np.array(np.load(data_path), dtype=float64)
    sentences = np.load(sents_path)

    #SET ME
    nb_sentences2extract = int((len(sentences) / 3)+1)

    test_outputs = rbm.test_rbm(dataset=data, learning_rate=0.000008, training_epochs=5, batch_size=4, n_hidden=9)

    features_sum = []
    for i in range(len(np.sum(test_outputs, axis=1))):
        features_sum.append([np.sum(test_outputs, axis=1)[i], i])


    index = [elt[1] for elt in sorted(features_sum, key=itemgetter(0), reverse=True)][:nb_sentences2extract]

    extracted_sentences = [(sentences[i], i) for i in index]

    extracted_sentences.sort(key=lambda x: x[1])

    summary = "".join([sent[0] for sent in extracted_sentences])

    summary_path = '/'.join(data_path.split('/')[:-1]) + '/summary.txt'

    with open(summary_path, "w") as output_file:
        output_file.write(summary)
    print('Summary ready :)')
    
    return summary_path

