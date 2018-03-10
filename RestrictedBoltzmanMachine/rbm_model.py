import numpy as np
# from theano.scalar import float32
import rbm
# from operator import itemgetter

mat1 = [ [0.1667, 1.0, 24, 1, 8, 0.0, 8, 1.8188, 1.0],
         [0.1429, -0.9529, 14, 0, 4, 0.0, 4, 0.973, 0.0472],
         [0.3636, 1.0, 22, 0, 9, 0.0, 9, 1.5455, 0.1929],
         [0.0, -0.9529, 14, 0, 6, 0.0, 6, 0.6931, 0.0],
         [0.1111, 0.8159, 9, 0, 4, 0.0, 4, 0.8047, 0.0589],
         [0.0952, -0.602, 21, 0, 8, 0.0, 8, 1.354, 0.0377],
         [0.0526, 0.3314, 19, 0, 6, 0.0, 6, 1.1513, 0.0811],
         [0.0909, -0.0295, 33, 1, 9, 0.0, 9, 1.354, 0.0312],
         [0.1333, -0.2752, 15, 1, 5, 0.0, 5, 0.8047, 0.0],
         [0.25, 0.5539, 8, 0, 5, 0.0, 5, 0.8959, 0.0],
         [0.1538, -0.7804, 13, 0, 4, 0.0, 4, 1.1513, 0.049],
         [0.3333, 0.9333, 12, 0, 6, 0.0, 6, 1.629, 0.378],
         [0.0, -0.9983, 15, 0, 8, 0.0, 8, 0.0, 0.0],
         [0.0769, 0.9691, 13, 0, 9, 0.0, 9, 0.3466, 0.0],
         [0.0, 1.0, 8, 1, 3, 0.0, 3, 0.0, 0.0]
         ]
# mat = np.load("my_matrix.dat.npy")
# mat1=np.array(mat,dtype=float32)
sentences=np.load("sentences.sav.npy")

temp = rbm.test_rbm(dataset=mat1, learning_rate=0.00008, training_epochs=5, batch_size=4,n_chains=4,n_hidden=9)

# print("\n\n")
# print(np.sum(temp, axis=1))

# enhanced_feature_sum = []
# enhanced_feature_sum2 = []

# for i in range(len(np.sum(temp, axis=1))):
#     enhanced_feature_sum.append([np.sum(temp, axis=1)[i], i])
#     enhanced_feature_sum2.append(np.sum(temp, axis=1)[i])

# print(enhanced_feature_sum)
# print("\n\n\n")

# Enh=sorted(enhanced_feature_sum, key=itemgetter(0),reverse=True)

# length_to_be_extracted = int((len(enhanced_feature_sum) / 3)+1)


# print("\n\nThe text is : \n\n")
# for x in range(len(sentences)):
#     print(sentences[x])

# print("\n\n\nExtracted sentences : \n\n\n")
# extracted_sentences = []


# indeces_extracted = []



# for x in range(0,length_to_be_extracted):
#     extracted_sentences.append([sentences[Enh[x][1]], Enh[x][1]])
#     indeces_extracted.append(enhanced_feature_sum[x][1])



# extracted_sentences.sort(key=lambda x: x[1])


# finalText = ""
# print("\n\n\nExtracted Final Text : \n\n\n")

# for i in range(len(extracted_sentences)):
#     print("\n" + extracted_sentences[i][0])
#     finalText = finalText + extracted_sentences[i][0]

# print(finalText)