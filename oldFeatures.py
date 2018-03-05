
##~~~~~~ Old Features ~~~~~~##

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
