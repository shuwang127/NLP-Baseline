# Author: Julia Jeng, Shu Wang, Arman Anwar
# Description: AIT 726 Homework 1
# Command to run the file:
#

import os
import re
import nltk

# The main function.
def main():
    print("-- AIT726 Homework 1 from Julia Jeng, Shu Wang, and Arman Anwar --")
    CreateVocabulary()


def CreateVocabulary():
    def Preprocess(data):
        data = re.sub(r'[a-z]*[:.]+\S+', '', data)  # remove url
        data = re.sub(r'\d+', '', data)  # remove numbers.
        def LowerFunc(matched):
            return matched.group(1).lower()
        data = re.sub('([A-Z][a-z]+)', LowerFunc, data)
        return data
    # if there is no 'tmp' folder, create one.
    if not os.path.exists('./tmp/'):
        os.mkdir('./tmp/')
    # read the training data.
    labelTrain = []
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
            data = Preprocess(data)
            print(data)

    print('There are %d/%d positive/negative samples.' % (sum(labelTrain), len(labelTrain)-sum(labelTrain)))

    return

# The program entrance.
if __name__ == "__main__":
    main()

