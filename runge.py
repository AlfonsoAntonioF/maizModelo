import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filePath = 'C:/Users/Usuario/Desktop/maizmodelo/Dataset/Ajustados2.csv'
dataSet = pd.read_csv(filePath)

tiempo_observado = dataSet['DIAS']
Precipitacion = dataSet['PRECIP']
Evaporacion = dataSet['EVAP']
Tmax = dataSet['TMAX']
Tmin = dataSet['TMIN']
poblacion_observada = dataSet['IAF']
# Modelo logístico con capacidad de carga variable
def logistico(P, Humedad, r, K0, w1):
    K = K0 + w1 * Humedad
    dPdt = r * P * (1 - P / K)
    return dPdt

# Implementación del método de Runge-Kutta de cuarto orden
def runge_kutta(h, P, Humedad, r, K0, w1):
    k1 = h * logistico(P, Humedad, r, K0, w1)
    k2 = h * logistico(P + 0.5 * k1, Humedad, r, K0, w1)
    k3 = h * logistico(P + 0.5 * k2, Humedad, r, K0, w1)
    k4 = h * logistico(P + k3, Humedad, r, K0, w1)
    return P + (k1 + 2*k2 + 2*k3 + k4) / 6

# Simulación del modelo usando Runge-Kutta
def simulate_model(r, K0, w1, P0, Humedad, h, num_steps):
    P = poblacion_observada
    P[0] = P0
    for i in range(1, num_steps):
        P[i] = runge_kutta(h, P[i-1], Humedad[i-1], r, K0, w1)
    return P

# Parámetros del modelo
r = 0.1  # tasa intrínseca de crecimiento
K0 = 12  # capacidad de carga básica
w1 = 0.1  # coeficiente para la humedad
P0 = 1  # población inicial

# Datos de humedad (puedes reemplazar estos datos con tus propios datos)
Humedad = Precipitacion-Evaporacion

# Configuración de la simulación
h = 1  # tamaño del paso
num_steps = len(Humedad)

# Simulación del modelo
poblacion_simulada = simulate_model(r, K0, w1, P0, Humedad, h, num_steps)
print(poblacion_simulada)

# Graficar los resultados
plt.plot(poblacion_simulada, label='Población simulada')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.legend()
plt.show()
