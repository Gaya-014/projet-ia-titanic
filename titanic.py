# Projet Titanic - je predis si un passager a survecu ou pas
# a partir de son age, sa classe, son sexe etc
# resultat : 80.45% de precision avec un RandomForest

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv("train.csv")

#print(data.head())
print(data.info())

# il manque des ages et des embarked, je les remplis
# median = valeur du milieu (pour des chiffres comme Age), mode = valeur la plus frequente (pour du texte comme Embarked)
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

# faut transformer sex et embarked en nombres pour que le modele comprenne
data = pd.get_dummies(data, columns=["Sex", "Embarked"])

X = data.drop(columns=["Survived", "Name", "Ticket", "Cabin", "PassengerId"])
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modele = RandomForestClassifier(n_estimators=100, random_state=42)
modele.fit(X_train, y_train)

predictions = modele.predict(X_test)
precision = accuracy_score(y_test, predictions)
print(f"Précision : {precision * 100:.2f}%")