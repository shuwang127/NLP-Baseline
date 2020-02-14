# Author: Julia Jeng, Shu Wang, Arman Anwar
# Description: AIT 726 Homework 1
# Command to run the file:
# python naive_bayes.py

import os
import re
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from itertools import chain

# The main function.
def main():
    print("-- AIT726 Homework 1 from Julia Jeng, Shu Wang, and Arman Anwar --")
    CreateVocabulary()

    # debug following
    dataset = 'Train'
    train = np.load('./tmp/'+dataset+'.npz', allow_pickle=True)
    lstem = 0
    if 1 == lstem:
        data = train['dataTrainStem'][0]
    else:
        data = train['dataTrain'][0]
    print(data)


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
    np.savez('tmp/Train.npz', labelTrain=labelTrain, dataTrain=dataTrain, dataTrainStem=dataTrainStem)
    # build the vocabulary from training set.
    vocab = list(set(list(chain.from_iterable(dataTrain))))
    vocabStem = list(set(list(chain.from_iterable(dataTrainStem))))
    print('Vocabulary: %d items.' % len(vocab))
    print('Vocabulary (stem): %d items.' % len(vocabStem))
    np.savez('tmp/Vocab.npz', vocab=vocab, vocabStem=vocabStem)
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
    np.savez('tmp/Test.npz', labelTest=labelTest, dataTest=dataTest, dataTestStem=dataTestStem)
    return

# The program entrance.
if __name__ == "__main__":
    main()

