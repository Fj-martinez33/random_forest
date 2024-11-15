# Librerias

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import json
from pickle import dump

#Obtenemos los datos

x_train_with_outliers = pd.read_csv("../data/processed/x_train_sel_with_outliers.csv")
x_train_without_outliers = pd.read_csv("../data/processed/x_train_sel_without_outliers.csv")
x_test_with_outliers = pd.read_csv("../data/processed/x_test_sel_with_outliers.csv")
x_test_without_outliers = pd.read_csv("../data/processed/x_test_sel_without_outliers.csv")
y_train = pd.read_csv("../data/processed/y_train.csv")
y_test = pd.read_csv("../data/processed/y_test.csv")

#Creamos dos listas para pasarlas por la funcion.

train_lst = [x_train_with_outliers, x_train_without_outliers]
test_lst = [x_test_with_outliers, x_test_without_outliers]

#Funcion de entrenamiento

def Training_model(x_trains, x_tests, y_train, y_test):
    results = []
    models = []

    for i in range(len(x_trains)):
        model = RandomForestClassifier(random_state=42)
        x_train = train_lst[i]
        model.fit(x_train, y_train)
    
        y_train_predict = model.predict(x_train)
        y_test_predict = model.predict(x_tests[i])

        result = {"index" : i, "Train Score" : accuracy_score(y_train, y_train_predict), "Test Score" : accuracy_score(y_test, y_test_predict)}

        models.append(model)
        results.append(result)
    
    return models, results

pre_models, pre_results = Training_model(train_lst, test_lst, y_train, y_test)

with open ("../data/processed/accuracy.json", "w") as j:
        json.dump( pre_results, j)

#Optimizacion de huperparametros

#Creamos el diccionario de hyperparametros

hyperparameters = {"n_estimators" : np.random.randint(100,size = 5 ), "criterion" : ["gini", "entropy", "log_loss"], "max_depth" : np.random.randint(7, size = 5), "min_samples_split" : np.random.randint( 10, size = 5), "min_samples_leaf" : np.random.randint( 7, size = 5), "random_state" : np.random.randint(1000, size = 5)}

grid = GridSearchCV(pre_models[1], hyperparameters, scoring="accuracy")
grid.fit(x_train_without_outliers, y_train)

clf = grid.best_estimator_
dump(clf, open(f"../models/rand_forest_model.sav", "wb"))

y_test_predict = clf.predict(x_test_without_outliers)

score = accuracy_score(y_test, y_test_predict)
