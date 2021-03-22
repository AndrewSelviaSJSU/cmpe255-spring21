import itertools

import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


class DiabetesClassifier:
  def __init__(self) -> None:
    self.columns = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    self.pima = pd.read_csv('diabetes.csv', header=0, names=self.columns, usecols=self.columns)
    print(self.pima.head())
    self.X_test = None
    self.y_test = None

  def define_feature(self, features, data=None):
    if data is None:
      data = self.pima
    X = data[features]
    y = data.label
    return X, y

  def train(self, X, y):
    X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, random_state=0)
    # train a logistic regression model on the training set
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    return logistic_regression

  def predict(self, X, y):
    model = self.train(X, y)
    y_pred_class = model.predict(self.X_test)
    return y_pred_class

  def calculate_accuracy(self, result):
    return metrics.accuracy_score(self.y_test, result)

  # def examine(self):
  #   dist = self.y_test.value_counts()
  #   print(dist)
  #   percent_of_ones = self.y_test.mean()
  #   percent_of_zeros = 1 - self.y_test.mean()
  #   return self.y_test.mean()

  def confusion_matrix(self, result):
    return metrics.confusion_matrix(self.y_test, result)


def baseline():
  classifier = DiabetesClassifier()
  X, y = classifier.define_feature(['pregnant', 'insulin', 'bmi', 'age'])
  result = classifier.predict(X, y)
  print(f"Prediction={result}")
  score = classifier.calculate_accuracy(result)
  print(f"score={score}")
  con_matrix = classifier.confusion_matrix(result)
  print(f"confusion_matrix={con_matrix}")


# https://stackoverflow.com/a/64320524/6073927
def create_power_set(iterable):
  for sl in itertools.product(*[[[], [i]] for i in iterable]):
    yield {j for i in sl for j in i}


def power_set_helper(feature_set, classifier, X, y):
  print(f"feature set: {feature_set}")
  X, y = classifier.define_feature(feature_set)
  result = classifier.predict(X, y)
  # print(f"Prediction={result}")
  score = classifier.calculate_accuracy(result)
  print(f"score={score}")
  con_matrix = classifier.confusion_matrix(result)
  print(f"confusion_matrix={con_matrix}")
  return feature_set, score


def solution_1():
  classifier = DiabetesClassifier()
  possible_features = classifier.columns
  possible_features.remove("label")
  X, y = classifier.define_feature(possible_features)
  power_set = list(create_power_set(possible_features))
  power_set.remove(set())
  scores = list(map(lambda feature_set: power_set_helper(feature_set, classifier, X, y), power_set))
  scores.sort(key=lambda x: x[1], reverse=True)
  print(scores)


def solution_4():
  classifier = DiabetesClassifier()
  best_features = ['pregnant', 'glucose', 'bp', 'bmi']
  X, y = classifier.define_feature(best_features)
  X = StandardScaler().fit_transform(X[best_features])
  result = classifier.predict(X, y)
  score = classifier.calculate_accuracy(result)
  print(f"score={score}")
  con_matrix = classifier.confusion_matrix(result)
  print(f"confusion_matrix={con_matrix}")


def solution_2():
  classifier = DiabetesClassifier()
  X, y = classifier.define_feature(['bp', 'bmi', 'pregnant', 'glucose'])
  # X, y = classifier.define_feature(['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age'])
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
  C_2d_range = [1e-2, 1, 1e2]
  gamma_2d_range = [1e-1, 1, 1e1]
  for C in C_2d_range:
    for gamma in gamma_2d_range:
      clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
      clf.fit(X_train, y_train)
      y_pred = clf.predict(X_test)
      print(f"C: {C}")
      print(f"gamma: {gamma}")
      print(metrics.accuracy_score(y_test, y_pred))


def solution_3():
  classifier = DiabetesClassifier()
  X, y = classifier.define_feature(['bp', 'bmi', 'pregnant', 'glucose'])
  # X, y = classifier.define_feature(['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age'])
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
  clf = svm.LinearSVC(max_iter=1000000)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  # con_matrix = clf.confusion_matrix(y_pred)
  print(metrics.accuracy_score(y_test, y_pred))
  # print(f"confusion_matrix={con_matrix}")


def solution_5():
  classifier = DiabetesClassifier()
  best_features = ['pregnant', 'glucose', 'bp', 'bmi']
  data = classifier.pima
  data = data[data.bp != 0]
  X, y = classifier.define_feature(best_features, data)
  X = StandardScaler().fit_transform(X[best_features])
  result = classifier.predict(X, y)
  score = classifier.calculate_accuracy(result)
  print(f"score={score}")
  con_matrix = classifier.confusion_matrix(result)
  print(f"confusion_matrix={con_matrix}")


def solution_6():
  classifier = DiabetesClassifier()
  best_features = ['pregnant', 'glucose', 'bp', 'bmi']
  data = classifier.pima
  data = data[data.glucose != 0]
  X, y = classifier.define_feature(best_features, data)
  X = StandardScaler().fit_transform(X[best_features])
  result = classifier.predict(X, y)
  score = classifier.calculate_accuracy(result)
  print(f"score={score}")
  con_matrix = classifier.confusion_matrix(result)
  print(f"confusion_matrix={con_matrix}")


def bucketize_pregnant(x):
  if x < 3:
    return 'few'
  else:
    return 'many'



def solution_7():
  classifier = DiabetesClassifier()
  best_features = ['pregnant', 'glucose', 'bp', 'bmi']

  data = classifier.pima
  data = data[data.glucose != 0]
  data = data[data.bp != 0]
  data = data[data.bmi != 0]

  bucketized = data.pregnant.map(bucketize_pregnant).to_frame()

  ohe = OneHotEncoder().fit_transform(bucketized).toarray()
  data['pregnant_few'] = ohe[:, 0]
  data['pregnant_many'] = ohe[:, 1]
  data = data.drop(columns=['pregnant'])
  new_features = ['pregnant_few', 'pregnant_many', 'glucose', 'bp', 'bmi']

  X, y = classifier.define_feature(new_features, data)
  X = StandardScaler().fit_transform(X[new_features])
  result = classifier.predict(X, y)
  score = classifier.calculate_accuracy(result)
  print(f"score={score}")
  con_matrix = classifier.confusion_matrix(result)
  print(f"confusion_matrix={con_matrix}")


if __name__ == "__main__":
  # baseline()
  # solution_1()
  # solution_2()
  # solution_3()
  # solution_4()
  # solution_5()
  # solution_6()
  solution_7()
