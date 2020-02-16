# Two baseline classifiers in Natural Language Processing

Load TrainSet: 1181/3000 positive/negative samples.

Vocabulary: 6666 items.

Vocabulary (stem): 5106 items.

Load TestSet: 1182/3000 positive/negative samples.

## Naive Bayes Classifier

naive bayes | noStem | freq

accuracy : 89.43%

naive bayes | noStem | bin

accuracy : 89.41%

naive bayes | noStem | tfidf

accuracy : 88.47%

naive bayes | Stem | freq

accuracy : 90.20%

naive bayes | Stem | bin

accuracy : 89.93%

naive bayes | Stem | tfidf

accuracy : 87.97%

## Logistic Regression

para: learningrate = 0.10, iternum = 30000

logistic | noReg | noStem | freq

accuracy : 90.00%

logistic | noReg | noStem | bin

accuracy : 89.69%

logistic | noReg | noStem | tfidf

accuracy : 90.17%

logistic | noReg | Stem | freq

accuracy : 89.89%

logistic | noReg | Stem | bin

accuracy : 90.03%

logistic | noReg | Stem | tfidf

accuracy : 90.15%

logistic | L2 | noStem | freq

accuracy : 90.05%

logistic | L2 | noStem | bin

accuracy : 89.50%

logistic | L2 | noStem | tfidf

accuracy : 90.17%

logistic | L2 | Stem | freq

accuracy : 90.27%

logistic | L2 | Stem | bin

accuracy : 89.91%

logistic | L2 | Stem | tfidf

accuracy : 89.43%

Thanks
