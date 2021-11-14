#!/usr/bin/env python3

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

dummycl = DummyClassifier(strategy="most_frequent")
gmb = GaussianNB()
dectree = tree.DecisionTreeClassifier()
rdforest = RandomForestClassifier()
logreg = LogisticRegression(solver="liblinear")
svc = svm.SVC(gamma='scale')

def accuracy_score(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        scores = cross_val_score(clf, X, y, cv=5)
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def confusion_matrix(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        predicted = cross_val_predict(clf, X, y, cv=5) 
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f" % metrics.accuracy_score(y, predicted))
        print(metrics.confusion_matrix(y, predicted))


data_test = pd.read_csv('data/bank-marketing/bank.test.csv', sep=';')
data_train = pd.read_csv('data/bank-marketing/bank.train.csv', sep=';')

# print(data_test)
# print(data_train)

# # le nombre d'attributs/colonnes
# print(len(data_test.columns), len(data_train.columns))

# # les premières valeurs et le type de chaque colonne
# for c in data_test.columns:
#     data_test.head()[c]
#     print()

# # les premières valeurs et le type de chaque colonne
# for c in data_train.columns:
#     data_train.head()[c]
#     print()

# print(data_test.isnull().sum().sum())
# print(data_train.isnull().sum().sum())


# Replace missing values by mean and scale numeric values
data_num = data_train.select_dtypes(include='number')
data_num = StandardScaler().fit_transform(data_num)


lst_classif = [dummycl, gmb, dectree, rdforest, logreg, svc]
lst_classif_names = ['Dummy', 'Naive Bayes', 'Decision tree', 'Random Forest', 'Logistic regression', 'SVM']
score = accuracy_score(lst_classif, lst_classif_names, data_num, data_train['y'])
matrix = confusion_matrix(lst_classif, lst_classif_names, data_num, data_train['y'])

# # Replace missing values by mean and discretize categorical values
# data_cat = data_train.select_dtypes(exclude='number').drop('y',axis=1)






