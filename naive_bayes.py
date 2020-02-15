# Author: Julia Jeng, Shu Wang, Arman Anwar
# Description: AIT 726 Homework 1
# Command to run the file:
# python naive_bayes.py

import os
import re
import math
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from itertools import chain

# The main function.
def main():
    print("-- AIT726 Homework 1 from Julia Jeng, Shu Wang, and Arman Anwar --")
    #CreateVocabulary()
    featTrain = ExtractFeatures('Train', 'noStem', 'freq')
    prior, likelihood = TrainNaiveBayes(featTrain)
    # debug following
    print(prior)
    print(likelihood)


# Read train/test sets and create vocabulary.
def CreateVocabulary():
    # pre-process the data.
    def Preprocess(data):
        # remove url
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        data = re.sub(pattern, '', data)
        # remove html special characters.
        pattern = r'&[(amp)(gt)(lt)]+;'
        data = re.sub(pattern, '', data)
        # remove independent numbers.
        pattern = r' \d+ '
        data = re.sub(pattern, ' ', data)
        # lower case capitalized words.
        pattern = r'([A-Z][a-z]+)'
        def LowerFunc(matched):
            return matched.group(1).lower()
        data = re.sub(pattern, LowerFunc, data)
        # remove hashtags.
        pattern = r'[@#]([A-Za-z]+)'
        data = re.sub(pattern, '', data)
        return data
    # get tokens.
    def GetTokens(data):
        # use tweet tokenizer.
        tknzr = TweetTokenizer()
        tokens = tknzr.tokenize(data)
        # tokenize at each punctuation.
        pattern = r'[A-Za-z]+\'[A-Za-z]+'
        for tk in tokens:
            if re.match(pattern, tk):
                subtokens = word_tokenize(tk)
                tokens.remove(tk)
                tokens = tokens + subtokens
        return tokens
    # process tokens with stemming.
    def WithStem(tokens):
        porter = PorterStemmer()
        tokensStem = []
        for tk in tokens:
            tokensStem.append(porter.stem(tk))
        return tokensStem
    # if there is no 'tmp' folder, create one.
    if not os.path.exists('./tmp/'):
        os.mkdir('./tmp/')
    # read the training data.
    labelTrain = []
    dataTrain = []
    dataTrainStem = []
    for root, ds, fs in os.walk('./tweet/train/'):
        for file in fs:
            fullname = os.path.join(root, file)
            # get the training label.
            if "positive" in fullname:
                label = 1
            else: # "negative" in fullname
                label = 0
            labelTrain.append(label)
            # get the training data.
            data = open(fullname, encoding="utf8").read()
            # print(data)
            # preprocess the data.
            data = Preprocess(data)
            # print(data)
            # get the tokens for the data.
            tokens = GetTokens(data)
            dataTrain.append(tokens)
            # print(tokens)
            # get the stemmed tokens for the data.
            tokensStem = WithStem(tokens)
            dataTrainStem.append(tokensStem)
            # print(tokensStem)
    print('Load TrainSet: %d/%d positive/negative samples.' % (sum(labelTrain), len(labelTrain)-sum(labelTrain)))
    np.savez('tmp/Train.npz', labelTrain = labelTrain, dataTrain = dataTrain, dataTrainStem = dataTrainStem)
    # build the vocabulary from training set.
    vocab = list(set(list(chain.from_iterable(dataTrain))))
    vocabStem = list(set(list(chain.from_iterable(dataTrainStem))))
    print('Vocabulary: %d items.' % len(vocab))
    print('Vocabulary (stem): %d items.' % len(vocabStem))
    np.savez('tmp/Vocab.npz', vocab = vocab, vocabStem = vocabStem)
    # read the testing data.
    labelTest = []
    dataTest = []
    dataTestStem = []
    for root, ds, fs in os.walk('./tweet/test/'):
        for file in fs:
            fullname = os.path.join(root, file)
            # get the testing label.
            if "positive" in fullname:
                label = 1
            else: # "negative" in fullname
                label = 0
            labelTest.append(label)
            # get the testing data.
            data = open(fullname, encoding="utf8").read()
            # print(data)
            # preprocess the data.
            data = Preprocess(data)
            # print(data)
            # get the tokens for the data.
            tokens = GetTokens(data)
            dataTest.append(tokens)
            # print(tokens)
            # get the stemmed tokens for the data.
            tokensStem = WithStem(tokens)
            dataTestStem.append(tokensStem)
            # print(tokensStem)
    print('Load TestSet: %d/%d positive/negative samples.' % (sum(labelTest), len(labelTest)-sum(labelTest)))
    np.savez('tmp/Test.npz', labelTest = labelTest, dataTest = dataTest, dataTestStem = dataTestStem)
    return

# extract features for a 'dataset' with or without 'stem' using 'method'
def ExtractFeatures(dataset = 'Train', lStem = 'noStem', method = 'freq'):
    # input validation.
    if dataset not in ['Train', 'Test']:
        print('Error: dataset input invalid!')
        return
    if lStem not in ['noStem', 'Stem']:
        print('Error: stem setting invalid!')
        return
    if method not in ['freq', 'bin', 'tfidf']:
        print('Error: method setting invalid!')
        return
    # sparse the corresponding dataset.
    dset = np.load('./tmp/' + dataset + '.npz', allow_pickle = True)
    if 'Stem' == lStem:
        data = dset['data' + dataset + lStem]
    else:
        data = dset['data' + dataset]
    D = len(data)
    # sparse the corresponding vocabulary.
    vset = np.load('./tmp/Vocab.npz', allow_pickle = True)
    if 'Stem' == lStem:
        vocab = vset['vocab' + lStem]
    else:
        vocab = vset['vocab']
    V = len(vocab)
    vocabDict = dict(zip(vocab, range(V)))
    # print(vocabDict)
    # get the feature matrix (freq).
    if 'freq' == method:
        features = np.zeros((D, V))
        index = 0
        for doc in data:
            for item in doc:
                if item in vocabDict:
                    features[index, vocabDict[item]] += 1
            index += 1
        return features
    # get the feature matrix (bin).
    if 'bin' == method:
        features = np.zeros((D, V))
        index = 0
        for doc in data:
            for item in doc:
                if item in vocabDict:
                    features[index, vocabDict[item]] = 1
            index += 1
        return features
    # get the feature matrix (tfidf):
    if 'tfidf' == method:
        return
    return

# train the naive bayes model.
def TrainNaiveBayes(features):
    # define the log prior.
    def GetLogPrior(labelTrain):
        # count the number.
        nDoc = len(labelTrain)
        nPos = list(labelTrain).count(1)
        nNag = list(labelTrain).count(0)
        # calculate the logprior.
        priorPos = math.log(nPos / nDoc)
        priorNag = math.log(nNag / nDoc)
        prior = [priorNag, priorPos]
        return prior
    # define loglikelihood.
    def GetLogLikelihood(features, labelTrain):
        # get V and D.
        V = len(features[0])
        D = len(features)
        cls = 2
        # initilaze likelihood matrix.
        likelihood = np.zeros((cls, V))
        for ind in range(D):
            for i in range(V):
                likelihood[labelTrain[ind]][i] += features[ind][i]
        # Laplace smoothing.
        denom = np.zeros((cls, 1))
        for lb in range(cls):
            denom[lb] = sum(likelihood[lb]) + V
            for i in range(V):
                likelihood[lb][i] += 1
                likelihood[lb][i] /= denom[lb]
                likelihood[lb][i] = math.log(likelihood[lb][i])
        return likelihood
    # sparse the corresponding label.
    dset = np.load('./tmp/Train.npz', allow_pickle = True)
    labelTrain = dset['labelTrain']
    # get the log prior.
    prior = GetLogPrior(labelTrain)
    # get the log likelihood
    likelihood = GetLogLikelihood(features, labelTrain)
    return prior, likelihood

# The program entrance.
if __name__ == "__main__":
    main()
