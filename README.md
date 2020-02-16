# Two baseline classifiers in Natural Language Processing

Load TrainSet: 1181/3000 positive/negative samples.

Vocabulary: 6666 items.

Vocabulary (stem): 5106 items.

Load TestSet: 1182/3000 positive/negative samples.

## Naive Bayes Classifier

|        | noStem | Stem   |
| ------ | ------ | ------ |
| freq   | 89.43% | 90.20% |
| bin    | 89.41% | 89.93% |
| tfidf  | 88.47% | 87.97% |

## Logistic Regression

parameters: learning rate = 0.10, iternum = 30000

Without regularization:
|        | noStem | Stem   |
| ------ | ------ | ------ |
| freq   | 90.00% | 89.89% |
| bin    | 89.69% | 90.03% |
| tfidf  | 90.17% | 90.15% |


With L2 regularization:
|        | noStem | Stem   |
| ------ | ------ | ------ |
| freq   | 90.05% | 90.27% |
| bin    | 89.50% | 89.91% |
| tfidf  | 90.17% | 89.43% |


Thanks
