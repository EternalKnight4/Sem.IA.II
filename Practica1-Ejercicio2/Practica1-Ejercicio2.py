# Universidad de Guadalajara
# Centro Universitario de Ciencias Exactas e Ingenierías
# División de Tecnologías para la Integración Ciber-Humana
# 
# Seminario de Solución de Problemas de Inteligencia Artificial II
# Ingenieria en Computación
# Sección D05
# 
# Práctica 1 - Ejercicio 2
# 
# Jorge Alberto Carrillo Partida / 216439258 / jorge.carrillo4392@alumnos.udg.mx
# 
# M. Diego Campos - 04 de Marzo de 2024

# ==============================================================================

# Librerías necesarias para el funcionamiento del programa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Clase Perceptron personalizada
class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate # Tasa de aprendizanje
        self.epochs = epochs                # Número de epocas
        self.weights = None                 # Inicialización de pesos
        self.bias = 0                       # Inicialización de sesgo

    # Función de activación
    def activation(self, x):
        return np.where(x >= 0, 1, -1)

    # Método para entrenar el perceptron
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation(linear_output)
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    # Método para realizar predicciones sobre los nuevos datos
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)

# Función para entrenar y evaluar el perceptrón simple
def train_perceptron(data, n_splits=5, train_size=0.8, learning_rate=0.01, epochs=10):
    results = []
    for _ in range(n_splits):
        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], train_size=train_size, random_state=np.random.randint(1000))
        
        # Crear y entrenar el perceptrón
        perceptron = Perceptron(learning_rate=learning_rate, epochs=epochs)
        perceptron.fit(X_train.values, y_train.values)
        
        # Evaluar el perceptrón
        y_pred = perceptron.predict(X_test.values)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Guardar los resultados
        results.append(accuracy)
    
    return results

# Cargar los datos de los documentos de la práctica
data_1d10 = pd.read_csv('spheres1d10.csv')
data_2d10 = pd.read_csv('spheres2d10.csv')
data_2d50 = pd.read_csv('spheres2d50.csv')
data_2d70 = pd.read_csv('spheres2d70.csv')

# Entrenar y evaluar el perceptrón personalizado con los datos spheres1d10
accuracies_1d10 = train_perceptron(data_1d10, n_splits=5, learning_rate=0.01, epochs=10)
print(f'Accuracies for spheres1d10.csv: {accuracies_1d10}')

# Entrenar y evaluar el perceptrón personalizado con los datos modificados spheres2d10, spheres2d50 y spheres2d70
accuracies_2d10 = train_perceptron(data_2d10, n_splits=10, learning_rate=0.01, epochs=10)
print(f'Accuracies for spheres2d10.csv: {accuracies_2d10}')

accuracies_2d50 = train_perceptron(data_2d50, n_splits=10, learning_rate=0.01, epochs=10)
print(f'Accuracies for spheres2d50.csv: {accuracies_2d50}')

accuracies_2d70 = train_perceptron(data_2d70, n_splits=10, learning_rate=0.01, epochs=10)
print(f'Accuracies for spheres2d70.csv: {accuracies_2d70}')
