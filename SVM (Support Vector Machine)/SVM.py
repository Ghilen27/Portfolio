import numpy as np
import pandas as pd
from numpy import mean
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# 1. Import données
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 2. Dimensions
print(np.shape(X)) # 150x4
print(np.shape(y)) # 150x.
print("---------------------------------------------------------------------------")

# 3. Conserver les classes 1 et 2
X = X[y != 0, :2]
y = y[y != 0]

# 4. Séparer les données train 50% et test 50%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, shuffle=True) # shuffle=True pour aleatoire

# 5. SVM linéaire vs SVM noyau polynomial
my_svm_l = SVC(kernel='linear')
my_svm_l.fit(X_train, y_train)
y_pred_l = my_svm_l.predict(X_test)
print("accuracy test linear:",round(accuracy_score(y_train, y_pred_l)*100,2),"%")
print("---------------------------------------------------------------------------")

my_svm_p = SVC(C=0.01, kernel='poly', degree=4, gamma=2)
my_svm_p.fit(X_train, y_train)
y_pred_p = my_svm_p.predict(X_test)
print("accuracy test poly:",round(accuracy_score(y_test, y_pred_p)*100,2),"%")
print("---------------------------------------------------------------------------")

# 6. Evaluer impact du choix du noyau et du paramètre de régularisation C
poly_05 = SVC(C=0.5, kernel='poly', degree=3, gamma=2)
poly_2 = SVC(C=2.0, kernel='poly', degree=3, gamma=2)
poly_05.fit(X_train, y_train)
poly_2.fit(X_train, y_train)
y_pred_05 = poly_05.predict(X_test)
y_pred_2 = poly_2.predict(X_test)
print("accuracy test poly C=0.5:",round(accuracy_score(y_test, y_pred_05)*100,2),"%")
print("accuracy test poly C=2.0:",round(accuracy_score(y_test, y_pred_2)*100,2),"%")
print("---------------------------------------------------------------------------")

# 7. Validation croisée
scores = cross_val_score(my_svm_l, X, y, cv=5) # données par sur test
print("accuracy cross validation", mean(scores)*100,"%")

