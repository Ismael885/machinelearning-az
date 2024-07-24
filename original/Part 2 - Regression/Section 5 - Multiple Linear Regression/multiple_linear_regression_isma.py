# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

# Regresion Lineal Multiple

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

"""ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)"""

onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
#X = np.array(ct.fit_transform(X), dtype=np.float)


# Evitar la trampa de las variables 
X = X[:, 1:]        # Eliminamos la primer columna para evitar la multicolinealidad

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Escalado de variables
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Ajustar el modelo de regresion lineal multiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Prediccion de los resultados en el conjutno de testing
y_pred = regression.predict(X_test)


# Construir el modelo optimo de RLM utilizando la Eliminacion Hacia Atras
import statsmodels.formula.api as sm
X = np.append(arr =np.ones((50,1)).astype(int), values = X, axis = 1)      # axis = 1 los agrega como columna
SL = 0.05

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()    # endog variable a predecir, exog variables indep con columnda de unos para term indep.
regression_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()    # endog variable a predecir, exog variables indep con columnda de unos para term indep.
regression_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()    # endog variable a predecir, exog variables indep con columnda de unos para term indep.
regression_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()    # endog variable a predecir, exog variables indep con columnda de unos para term indep.
regression_OLS.summary()

X_opt = X[:, [0, 3]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()    # endog variable a predecir, exog variables indep con columnda de unos para term indep.
regression_OLS.summary()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

regression.fit(X_train[:, [0, 3]], y_train)

y_pred = regression.predict(X_test[:, [0, 3]])
