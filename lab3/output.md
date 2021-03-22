# Lab 3

| Experiment | Accuracy |   Confusion Matrix    | Comment                                                                                  |
|------------|----------|-----------------------|------------------------------------------------------------------------------------------|
| Baseline   |    0.677 | [[114  16] [ 46  16]] | LogisticRegression; {'pregnant', 'insulin', 'bmi', 'age'}                                |
| Solution 1 |    0.812 | [[119  11] [ 25  37]] | LogisticRegression; {'glucose', 'pregnant', 'bp', 'bmi'}                                 |
| Solution 2 |    0.682 | N/A                   | SVM                                                                                      |
| Solution 3 |    0.812 | N/A                   | Linear SVC                                                                               |
| Solution 4 |    0.812 | [[119  11] [ 25  37]] | StandardScaler + LogisticRegression; {'glucose', 'pregnant', 'bp', 'bmi'}                |
| Solution 5 |    0.744 | [[105  12] [ 35  32]] | bp != 0 + StandardScaler + LogisticRegression; {'glucose', 'pregnant', 'bp', 'bmi'}      |
| Solution 6 |    0.754 | [[110  18] [ 29  34]] | glucose != 0 + StandardScaler + LogisticRegression; {'glucose', 'pregnant', 'bp', 'bmi'} |
| Solution 7 |    0.751 | [[95 26] [19 41]]     | One-Hot Encoding |

The approaches I have taken are broken down into sections below:

## Baseline

This is what was initially provided. It trains a LogisticRegression model with the following features: {'pregnant', 'insulin', 'bmi', 'age'}. It simply establishes a baseline accuracy of 0.812

## Solution 1

The first thing I tried was to iterate over the power set of all possible features. I trained a LogisticRegression model identically to the baseline but with each iterative feature set. The feature set which produced the highest accuracy (0.812) was: {'glucose', 'pregnant', 'bp', 'bmi'}. 

## Solution 2

Next, I wanted to try a SVM model, since I learned about them last semester and figured they might be effective. I used an rbf kernel and tried various hyper-parameters, but no result improved upon Solution 1. Unfortunately, SVMs also lack a confusion matrix function.

## Solution 3

After reading through the scikit-learn documentation on SVMs, I suspected a Linear SVC might be an interesting approach to take. It just matched Solution 1, though.

## Solution 4

I came back to LogisticRegression to try feature engineering approaches rather than different models. I used a StandardScaler, but it didn't make a difference on top of Solution 1.

## Solution 5

I tried to remove outliers where `bp` == 0, but it actually made the model perform worse.

## Solution 6

This time, I tried removing outliers where `glucose` == 0, but it also made the model perform worse.

## Solution 7

I tried to bucketize the pregnant feature into *few* and *many* divided by 3 births. Then, I one-hot encoded the feature. The accuracy went down, but perhaps with more attempts, I could make a bigger dent. Not enough time now, though.