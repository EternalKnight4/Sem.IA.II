# Universidad de Guadalajara
# Centro Universitario de Ciencias Exactas e Ingenierías
# División de Tecnologías para la Integración Ciber-Humana
# 
# Seminario de Solución de Problemas de Inteligencia Artificial II
# Ingenieria en Computación
# Sección D05
# 
# Práctica 1 - Ejercicio 3 - Parte2
# 
# Jorge Alberto Carrillo Partida / 216439258 / jorge.carrillo4392@alumnos.udg.mx
# 
# M. Diego Campos - 15 de Abril de 2024

# ==============================================================================

# Librerías necesarias para el funcionamiento del programa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l1

# Función para crear el perceptrón multicapa
def create_MLP(layers):
    model = Sequential([Input(shape=(2,))])
    for neurons in layers:
        model.add(Dense(neurons, activation='sigmoid', kernel_regularizer=l1(0.0001)))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Cargar los datos de los documentos de la práctica
data = pd.read_csv('concentlite.csv')
X = data[['x1', 'x2']].values
y = data['y'].values

# Dividir los datos y normalizar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configurar las capas de la red
layers = [18, 15, 8]

# Crear, entrenar y evaluar el perceptrón
model = create_MLP(layers)
optimizer = SGD(learning_rate=0.05, momentum=0.9)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train, epochs=500, batch_size=50, verbose=1, validation_split=0.2)

loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Accuracy: {accuracy}')

# Función para visualizar los resultados de clasificación en los datos de prueba
def plot_test_data(X, y, model):
    plt.figure(figsize=(10, 8))
    y_pred = model.predict(X) > 0.5
    # Clase real
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.5, marker='s', edgecolor='k', label='Actual Class')
    # Clase predicha
    plt.scatter(X[:, 0], X[:, 1], c=y_pred.flatten(), cmap='viridis', alpha=0.5, marker='x', label='Predicted Class')
    plt.title("Test Data Classification")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# Visualizar la clasificación en los datos de prueba
plot_test_data(X_test_scaled, y_test, model)
