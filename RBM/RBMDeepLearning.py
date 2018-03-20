import numpy as np
from theano.scalar import float64
import rbm
from operator import itemgetter
import os

#~~~~ Choose language ~~~~#
# lang = 'en'
lang = 'en'

if lang == 'en':
    matrix_name= "en/english.txt_dat.npy"
    sentences_name="en/english.txt_sents.npy"
else:
    matrix_name= "ar/arabic.txt_dat.npy"
    sentences_name="ar/arabic.txt_sents.npy"

mat = np.load(matrix_name)

mat1=np.array(mat,dtype=float64)
sentences=np.load(sentences_name)
temp = rbm.test_rbm(dataset=mat1, learning_rate=0.000008, training_epochs=5, batch_size=4,n_hidden=9)

print("\n\n")
print(np.sum(temp, axis=1))

enhanced_feature_sum = []
enhanced_feature_sum2 = []

for i in range(len(np.sum(temp, axis=1))):
    enhanced_feature_sum.append([np.sum(temp, axis=1)[i], i])
    enhanced_feature_sum2.append(np.sum(temp, axis=1)[i])

print(enhanced_feature_sum)
print("\n\n\n")

Enh=sorted(enhanced_feature_sum, key=itemgetter(0),reverse=True)

length_to_be_extracted = int((len(enhanced_feature_sum) / 3)+1)


print("\n\nThe text is : \n\n")
for x in range(len(sentences)):
    print(sentences[x])

print("\n\n\nExtracted sentences : \n\n\n")
extracted_sentences = []


indeces_extracted = []



for x in range(0,length_to_be_extracted):
    extracted_sentences.append([sentences[Enh[x][1]], Enh[x][1]])
    indeces_extracted.append(enhanced_feature_sum[x][1])



extracted_sentences.sort(key=lambda x: x[1])


finalText = ""
print("\n\n\nExtracted Final Text : \n\n\n")

for i in range(len(extracted_sentences)):
    print("\n" + extracted_sentences[i][0])
    finalText = finalText + extracted_sentences[i][0]

print(finalText)
f=open('summary.txt', "w")
f.write(finalText)
f.close()
