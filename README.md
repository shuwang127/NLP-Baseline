# Two baseline classifiers in Natural Language Processing

## Naive Bayes Classifier

Load TrainSet: 1181/3000 positive/negative samples.

Vocabulary: 6666 items.

Vocabulary (stem): 5106 items.

Load TestSet: 1182/3000 positive/negative samples.

-------------------------------------------
naive bayes | noStem | freq
accuracy : 89.43%
confusion matrix :      (actual)
                    Neg         Pos
(predicted) Neg     2896(TN)    338 (FN)
            Pos     104 (FP)    844 (TP)
-------------------------------------------
-------------------------------------------
naive bayes | noStem | bin
accuracy : 89.41%
confusion matrix :      (actual)
                    Neg         Pos
(predicted) Neg     2913(TN)    356 (FN)
            Pos     87  (FP)    826 (TP)
-------------------------------------------
-------------------------------------------
naive bayes | noStem | tfidf
accuracy : 88.47%
confusion matrix :      (actual)
                    Neg         Pos
(predicted) Neg     2789(TN)    271 (FN)
            Pos     211 (FP)    911 (TP)
-------------------------------------------
-------------------------------------------
naive bayes | Stem | freq
accuracy : 90.20%
confusion matrix :      (actual)
                    Neg         Pos
(predicted) Neg     2896(TN)    306 (FN)
            Pos     104 (FP)    876 (TP)
-------------------------------------------
-------------------------------------------
naive bayes | Stem | bin
accuracy : 89.93%
confusion matrix :      (actual)
                    Neg         Pos
(predicted) Neg     2904(TN)    325 (FN)
            Pos     96  (FP)    857 (TP)
-------------------------------------------
-------------------------------------------
naive bayes | Stem | tfidf
accuracy : 87.97%
confusion matrix :      (actual)
                    Neg         Pos
(predicted) Neg     2779(TN)    282 (FN)
            Pos     221 (FP)    900 (TP)
-------------------------------------------

## Logistic Regression
