# Universidad de Guadalajara
# Centro Universitario de Ciencias Exactas e Ingenierías
# División de Tecnologías para la Integración Ciber-Humana
# 
# Seminario de Solución de Problemas de Inteligencia Artificial II
# Ingenieria en Computación
# Sección D05
# 
# Práctica 1 - Ejercicio 4
# 
# Jorge Alberto Carrillo Partida / 216439258 / jorge.carrillo4392@alumnos.udg.mx
# 
# M. Diego Campos - 22 de Abril de 2024

# ==============================================================================

# Librerías necesarias para el funcionamiento del programa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

# Cargar los datos de los documentos de la práctica
file_path = 'irisbin.csv'
df = pd.read_csv(file_path)

# Transformar las etiquetas a una única etiqueta categórica
df['class'] = df.apply(lambda row: 0 if row['y1'] == -1 and row['y2'] == -1 and row['y3'] == 1 else 
                                 (1 if row['y1'] == -1 and row['y2'] == 1 and row['y3'] == -1 else 2), axis=1)

# Separar las características y la etiqueta transformada
X = df[['x1', 'x2', 'x3', 'x4']].values
y = df['class'].values

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir el dataset en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear y entrenar el perceptrón multicapa con parámetros ajustados
mlp = MLPClassifier(hidden_layer_sizes=(12,), activation='relu', solver='adam', max_iter=2000, random_state=42)
mlp.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de prueba
y_pred = mlp.predict(X_test)

# Generar el reporte de clasificación
print("Clasificación en el conjunto de prueba:")
print(classification_report(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica']))

# Validación Leave-One-Out
loo = LeaveOneOut()
loo_scores = cross_val_score(mlp, X_scaled, y, cv=loo, scoring='accuracy')
loo_mean = np.mean(loo_scores)
loo_std = np.std(loo_scores)

# Resultados de Leave-One-Out
print(f"Leave-One-Out Accuracy: {loo_mean:.2f}")
print(f"Leave-One-Out Std Dev: {loo_std:.2f}")

# Validación Leave-K-Out (Leave-5-Out como ejemplo)
kfold = KFold(n_splits=30)
kfold_scores = cross_val_score(mlp, X_scaled, y, cv=kfold, scoring='accuracy')
kfold_mean = np.mean(kfold_scores)
kfold_std = np.std(kfold_scores)

# Resultados de Leave-K-Out
print(f"Leave-K-Out Accuracy (k=5): {kfold_mean:.2f}")
print(f"Leave-K-Out Std Dev (k=5): {kfold_std:.2f}")
