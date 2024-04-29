# Universidad de Guadalajara
# Centro Universitario de Ciencias Exactas e Ingenierías
# División de Tecnologías para la Integración Ciber-Humana
# 
# Seminario de Solución de Problemas de Inteligencia Artificial II
# Ingenieria en Computación
# Sección D05
# 
# Práctica 1 - Ejercicio 1
# 
# Jorge Alberto Carrillo Partida / 216439258 / jorge.carrillo4392@alumnos.udg.mx
# 
# M. Diego Campos - 26 de Febrero de 2024

# ==============================================================================

# Librerías necesarias para el funcionamiento del programa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle

# Clase necesaria para la red neuronal
class Perceptron:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate  # Tasa de aprendizanje
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

    # Método para visualizar la frontera de decisión del perceptron
    def decision_boundary(self, X, y):
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', marker='o', label='Training data')
        x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        x2 = -(self.weights[0] * x1 + self.bias) / self.weights[1]
        ax.plot(x1, x2, 'k', label='Frontera de decisión')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.legend()
        plt.title('Frontera de decisión del Perceptron - XOR')
        plt.show()

# Cargar los datos de los documentos de la práctica
data_trn = pd.read_csv('XOR_trn.csv')
data_tst = pd.read_csv('XOR_tst.csv')

# Combinar, mezclar y dividir los datos para un entrenamiento con el 80% de datos y 20% para las pruebas
combined_X = np.vstack((data_trn[['X1', 'X2']].values, data_tst[['X1', 'X2']].values))
combined_y = np.hstack((data_trn['Y'].values, data_tst['Y'].values))
combined_X, combined_y = shuffle(combined_X, combined_y, random_state=1)

split_index = int(0.8 * len(combined_y))
X_train = combined_X[:split_index]
y_train = combined_y[:split_index]
X_test = combined_X[split_index:]
y_test = combined_y[split_index:]

# Crear, entrenar y evaluar el perceptrón
perceptron = Perceptron(learning_rate=0.1, epochs=10)
perceptron.fit(X_train, y_train)
predictions = perceptron.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Visualizar la frontera de decisión
perceptron.decision_boundary(X_train, y_train)
