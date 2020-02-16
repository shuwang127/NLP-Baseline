'''
  Author: Julia Jeng, Shu Wang, Arman Anwar
  Description: AIT 726 Homework 1
  Usage: Put file 'naive_bayes.py' and folder 'twitter' in the same folder.
  Command to run: 
      python naive_bayes.py
'''

import os
import re
import sys
import math
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from itertools import chain

# Logger: redirect the stream on screen and to file.
class Logger(object):
    def __init__(self, filename = "log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

# The main function.
def main():
    # initialize the log file.
    logPath = './naive_bayes.txt'
    if os.path.exists(logPath):
        os.remove(logPath)
    sys.stdout = Logger(logPath)

    print("-- AIT726 Homework 1 from Julia Jeng, Shu Wang, and Arman Anwar --")
    # create the vocabulary.
    CreateVocabulary()
    # run demo.
    DemoNaiveBayes('noStem', 'freq')
    DemoNaiveBayes('noStem', 'bin')
    DemoNaiveBayes('noStem', 'tfidf')
    DemoNaiveBayes('Stem', 'freq')
    DemoNaiveBayes('Stem', 'bin')
    DemoNaiveBayes('Stem', 'tfidf')
    return

# a demo of naive bayes classifier with different dataset and features.
def DemoNaiveBayes(lStem = 'noStem', method = 'freq'):
    '''
    a demo of naive bayes classifier with different dataset and features.
    :param lStem: stem setting - 'noStem', 'Stem'
    :param method: feature selection - 'freq', 'bin', 'tfidf'
    :return: none
    '''
    # input validation.
    if lStem not in ['noStem', 'Stem']:
        print('Error: stem setting invalid!')
        return
    if method not in ['freq', 'bin', 'tfidf']:
        print('Error: method setting invalid!')
        return

    # extract training features with 'method' on 'lStem' dataset.
    featTrain = ExtractFeatures('Train', lStem, method)
    # get the model parameters.
    prior, likelihood = TrainNaiveBayes(featTrain)
    # extract testing features with 'method' on 'lStem' dataset.
    featTest = ExtractFeatures('Test', lStem, method)
    # get testing predictions using model parameters.
    accuracy, confusion = TestNaiveBayes(prior, likelihood, featTest)
    # output the results on screen and to files.
    OutputNaiveBayes(accuracy, confusion, lStem, method)
    # debug
    return

# Read train/test sets and create vocabulary.
def CreateVocabulary():
    '''
    read train and test sets and create vocabulary.
    :return: none
    '''
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
    '''
    extract features for a 'dataset' with or without 'stem' using 'method'
    :param dataset: dataset type - 'Train', 'Test'
    :param lStem: stem setting - 'noStem', 'Stem'
    :param method: feature selection - 'freq', 'bin', 'tfidf'
    :return: features - D * V
    '''
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

    # get the feature matrix (freq).
    if 'freq' == method:
        features = np.zeros((D, V))
        ind = 0
        for doc in data:
            for item in doc:
                if item in vocabDict:
                    features[ind][vocabDict[item]] += 1
            ind += 1
        return features
    # get the feature matrix (bin).
    if 'bin' == method:
        features = np.zeros((D, V))
        ind = 0
        for doc in data:
            for item in doc:
                if item in vocabDict:
                    features[ind][vocabDict[item]] = 1
            ind += 1
        return features
    # get the feature matrix (tfidf):
    if 'tfidf' == method:
        # get freq and bin features.
        termFreq = np.zeros((D, V))
        termBin = np.zeros((D, V))
        for ind, doc in enumerate(data):
            for item in doc:
                if item in vocabDict:
                    termFreq[ind][vocabDict[item]] += 1
                    termBin[ind][vocabDict[item]] = 1
        # get tf (1+log10)
        tf = np.zeros((D, V))
        for ind in range(D):
            for i in range(V):
                if termFreq[ind][i] > 0:
                    tf[ind][i] = 1 + math.log(termFreq[ind][i], 10)
        del termFreq
        # find idf
        if 'Train' == dataset:
            # get df
            df = np.zeros((V, 1))
            for ind in range(D):
                for i in range(V):
                    df[i] += termBin[ind][i]
            # get idf (log10(D/df))
            idf = np.zeros((V, 1))
            for i in range(V):
                if df[i] > 0:
                    idf[i] = math.log(D, 10) - math.log(df[i], 10)
            del df
            np.save('./tmp/idf.npy', idf)
        else:
            # if 'Test' == dataset, get idf from arguments.
            idf = np.load('./tmp/idf.npy')
        del termBin
        # get tfidf
        tfidf = np.zeros((D, V))
        for ind in range(D):
            for i in range(V):
                tfidf[ind][i] = tf[ind][i] * idf[i]
        return tfidf
    return

# train the naive bayes model.
def TrainNaiveBayes(features):
    '''
    train the naive bayes model.
    :param features: training set features
    :return: model parameters - prior, likelihood
    '''
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

# test and evaluate the performance.
def TestNaiveBayes(prior, likelihood, featTest):
    '''
    test and evaluate the performance.
    :param prior: model parameter
    :param likelihood: model parameter
    :param featTest: testing data features
    :return: evaluations - accuracy, confusion
    '''
    # get predictions for testing samples with model parameters.
    def GetPredictions(prior, likelihood, featTest):
        # get V and D.
        V = len(featTest[0])
        D = len(featTest)
        cls = 2
        # get pred(D, cls) matrix and predictions(D, 1).
        pred = np.zeros((D, cls))
        predictions = np.zeros((D, 1))
        for ind in range(D):
            for lb in range(cls):
                pred[ind][lb] += prior[lb]
                for i in range(V):
                    pred[ind][lb] += likelihood[lb][i] * featTest[ind][i]
            predictions[ind] = list(pred[ind]).index(max(pred[ind]))
        return predictions

    # evaluate the predictions with gold labels, and get accuracy and confusion matrix.
    def Evaluation(predictions):
        # sparse the corresponding label.
        dset = np.load('./tmp/Test.npz', allow_pickle = True)
        labelTest = dset['labelTest']
        D = len(labelTest)
        cls = 2
        # get confusion matrix.
        confusion = np.zeros((cls, cls))
        for ind in range(D):
            nRow = int(predictions[ind])
            nCol = int(labelTest[ind])
            confusion[nRow][nCol] += 1
        # get accuracy.
        accuracy = 0
        for ind in range(cls):
            accuracy += confusion[ind][ind]
        accuracy /= D
        return accuracy, confusion

    # get predictions for testing samples.
    predictions = GetPredictions(prior, likelihood, featTest)
    # get accuracy and confusion matrix.
    accuracy, confusion = Evaluation(predictions)
    return accuracy, confusion

# output the results.
def OutputNaiveBayes(accuracy, confusion, lStem = 'noStem', method = 'freq'):
    '''
    output the results to screen and file.
    :param accuracy: accuracy - 1x1
    :param confusion: confusion - 2x2
    :param lStem: stem setting - 'noStem', 'Stem'
    :param method: feature selection - 'freq', 'bin', 'tfidf'
    :return: none
    '''
    # input validation.
    if lStem not in ['noStem', 'Stem']:
        print('Error: stem setting invalid!')
        return
    if method not in ['freq', 'bin', 'tfidf']:
        print('Error: method setting invalid!')
        return

    # output on screen and to file.
    print('-------------------------------------------')
    print('naive bayes | ' + lStem + ' | ' + method)
    print('accuracy : %.2f%%' % (accuracy * 100))
    print('confusion matrix :      (actual)')
    print('                    Neg         Pos')
    print('(predicted) Neg     %-4d(TN)    %-4d(FN)' % (confusion[0][0], confusion[0][1]))
    print('            Pos     %-4d(FP)    %-4d(TP)' % (confusion[1][0], confusion[1][1]))
    print('-------------------------------------------')
    return

# The program entrance.
if __name__ == "__main__":
    main()
