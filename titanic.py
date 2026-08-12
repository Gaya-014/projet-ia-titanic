import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# on charge les données
data = pd.read_csv("train.csv")

# on bouche les trous : âge manquant -> médiane, port manquant -> valeur la plus fréquente
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

# Sex et Embarked sont du texte, on les transforme en colonnes de 0/1
data = pd.get_dummies(data, columns=["Sex", "Embarked"])

# X = ce qui sert à prédire, y = ce qu'on veut prédire (survécu ou pas)
X = data.drop(columns=["Survived", "Name", "Ticket", "Cabin", "PassengerId"])
y = data["Survived"]

# 80% pour entraîner, 20% pour tester
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# on crée et on entraîne le modèle
modele = DecisionTreeClassifier(max_depth=4, random_state=42)
modele.fit(X_train, y_train)

# on teste sur les 20% jamais vus, et on mesure le résultat
predictions = modele.predict(X_test)
precision = accuracy_score(y_test, predictions)
print(f"Précision : {precision * 100:.2f}%")