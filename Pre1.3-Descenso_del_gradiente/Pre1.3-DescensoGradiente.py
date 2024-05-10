# Universidad de Guadalajara
# Centro Universitario de Ciencias Exactas e Ingenierías
# División de Tecnologías para la Integración Ciber-Humana
# 
# Seminario de Solución de Problemas de Inteligencia Artificial II
# Ingenieria en Computación
# Sección D05
# 
# Pre 1.3 - Descenso del gradiente
# 
# Jorge Alberto Carrillo Partida / 216439258 / jorge.carrillo4392@alumnos.udg.mx
# 
# M. Diego Campos - 22 de Marzo de 2024

# ==============================================================================

# Librerías necesarias para el funcionamiento del programa
import numpy as np
import matplotlib.pyplot as plt

# Definimos la función objetivo que queremos minimizar
def f(x1, x2):
    return 10 - np.exp(-(x1**2 + 3*x2**2))

# Calculamos el gradiente de la función en un punto dado
def gradient_f(x1, x2):
    # Derivadas parciales respecto a x1 y x2
    df_dx1 = 2 * x1 * np.exp(-(x1**2 + 3*x2**2))
    df_dx2 = 6 * x2 * np.exp(-(x1**2 + 3*x2**2))
    return np.array([df_dx1, df_dx2])

# Implementa el algoritmo de descenso del gradiente para encontrar el mínimo de la función
def gradient_descent(starting_point, learning_rate, iterations):
    x = starting_point
    trajectory = [x]
    values = [f(x[0], x[1])]
    
    for _ in range(iterations):
        grad = gradient_f(x[0], x[1])
        x = x - learning_rate * grad
        trajectory.append(x)
        values.append(f(x[0], x[1]))
        
    return x, values, trajectory

# Parámetros iniciales
learning_rate = 0.01
iterations = 100
starting_point = np.random.uniform(-1, 1, 2)

# Ejecutando el descenso del gradiente
optimal_x, values, trajectory = gradient_descent(starting_point, learning_rate, iterations)

# Mostrando los resultados
print(f"Valores óptimos de x1, x2: {optimal_x}")
print(f"Valor final de la función: {values[-1]}")

# Graficando la convergencia del valor de la función
plt.figure(figsize=(10, 6))
plt.plot(values)
plt.title("Convergencia del valor de la función")
plt.xlabel("Iteración")
plt.ylabel("Valor de la función")
plt.grid(True)
plt.show()
