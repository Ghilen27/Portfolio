import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

### Perceptron ###

def predict_class(row,w):
    return w[0]+np.dot(w[1:len(w)],row)

def Perceptronn(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    nmiss_list = []
    for epoch in range(n_epoch):
        nmiss = 0
        for row in train:
            prediction = predict_class(row[0:len(row) - 1], weights)
            error = row[-1] - prediction
            if error != 0:
                weights[0] = weights[0] + l_rate*row[-1]
                for i in range(len(row) - 1):
                    weights[i + 1] = weights[i + 1] + l_rate*row[-1]*row[i]
                nmiss = nmiss + 1
        nmiss_list.append(nmiss)
    print("> epoch = %d, lrate = %.3f, accuracy = %.3f" % (epoch, l_rate, 1 - nmiss_list[-1]/len(train)))
    return weights, nmiss_list, 1 - nmiss_list[-1]/len(train)


### Logistic regression ###

# 2
data = pd.read_csv("banking.csv") #Importer le fichier csv
data = data.dropna()


# 3
print(data.head()) # 41188 observations et 21 variables
print(data.info())
# Les variables d'entrées sont les 20 premieres
# La variable de sortie est y
print("---------------------------------------------------------------------------")


# 4
# On supprime les colonnes
data_biss = data.drop(columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome'])
print(data_biss.info())
print("---------------------------------------------------------------------------")


# 5
my_log = linear_model.LogisticRegression()


# 6
data_X = data_biss.iloc[:,0:10]
scaler = StandardScaler()
data_X_scaled = scaler.fit_transform(data_X) # Normalisé X
data_Y = data_biss['y']
my_log.fit(data_X_scaled, data_Y)

# pour savoir si les donnes sont equilibrer ou pas "le nombre de 0 et 1"
class_distribution = data_Y.value_counts()
class_percentage = class_distribution / len(data_Y) * 100

print("Nombre de 0 et de 1 :")
print(class_distribution)
print("---------------------------------------------------------------------------")

print("\nPourcentage de 0 et 1 :")
print(class_percentage)
print("---------------------------------------------------------------------------")

# Visualiser la distribution des classes
plt.figure(figsize=(8, 6))
class_distribution.plot(kind='bar')
plt.title('Distribution des classes')
plt.show()


# 7
print(f"le coef est {my_log.coef_}")
print(f"intercept est {my_log.intercept_}")
print("---------------------------------------------------------------------------")


# 8
predict = my_log.predict(data_X_scaled)
print("Prediction:", predict)

data_Y_proba = my_log.predict_proba(data_X_scaled)[:,:]
sns.histplot(data_Y_proba) # les probas sont concentrée autour de 0 et 1, donc le modèle fait des prédictions relativement sûres.
plt.show()

score = my_log.score(data_X_scaled, data_Y)
print("Score:", np.round(score*100, 2),"%")
print("---------------------------------------------------------------------------")


# 10
X_train, X_test, y_train, y_test = train_test_split(data_X_scaled, data_Y, test_size=0.20, random_state=42)


# 11
ppn = Perceptron(eta0=0.15, random_state=0)
my_log.fit(X_test, y_test)
ppn.fit(X_test, y_test)


# 12
# Test:
y_pred_test = my_log.predict(X_test)
y_pred_test_ppn = my_log.predict(X_test)

print("f1 score test:",round(f1_score(y_test, y_pred_test)*100,2),"%") # UTILE quand les donnees sont pas equilibrees et je veux minimiser les FP et FN
print("matrice de confusion:",confusion_matrix(y_test, y_pred_test))
print("Classification", classification_report(y_test, y_pred_test))

print("---------------------------------------------------------------------------")

print("PPN f1 score test:",round(f1_score(y_test, y_pred_test_ppn)*100,2),"%")
print("PPN matrice de confusion:",confusion_matrix(y_test, y_pred_test_ppn))
print("PPN Classification", classification_report(y_test, y_pred_test_ppn))


print("---------------------------------------------------------------------------")

# 13 validation croisé

cv = KFold(n_splits=10, shuffle=False)
scores = cross_val_score(my_log, data_X_scaled, data_Y, scoring="f1", cv=cv, n_jobs=-1)
print("logistic regression f1 score", (round(mean(scores*100),2), round(std(scores*100),2)),"%")
