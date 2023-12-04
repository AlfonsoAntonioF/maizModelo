import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filePath = 'C:/Users/Usuario/Desktop/maizmodelo/Dataset/Ajustados2.csv'
dataSet = pd.read_csv(filePath)

tiempo_observado = dataSet['DIAS']
Precipitacion = dataSet['PRECIP']
Evaporacion = dataSet['EVAP']
Tmax = dataSet['TMAX']
Tmin = dataSet['TMIN']
poblacion_observada = dataSet['IAF']

Humedad = Precipitacion-Evaporacion
def f(x):
    F=1/(4.20*np.sqrt(2*np.pi))*np.exp(-1*(x+0.34)**2/2*(4.20**2))
    return F
# Modelo con capacidad de carga variable y tasa de crecimiento potencial
def logistic_growth(P, Humedad, r, K0, w1, c):
    K = K0 + w1 * Humedad
    dPdt = r * (P**c) * (1 - P / K)
    return dPdt

# Implementación del método de Runge-Kutta de cuarto orden
def runge_kutta(h, P, Humedad, r, K0, w1, c):
    k1 = h * logistic_growth(P, Humedad, r, K0, w1, c)
    k2 = h * logistic_growth(P + 0.5 * k1, Humedad, r, K0, w1, c)
    k3 = h * logistic_growth(P + 0.5 * k2, Humedad, r, K0, w1, c)
    k4 = h * logistic_growth(P + k3, Humedad, r, K0, w1, c)
    return P + (k1 + 2*k2 + 2*k3 + k4) / 6

# Simulación del modelo usando Runge-Kutta
def simulate_model(r, K0, w1, c, P0, Humedad, h, num_steps):
    P = np.zeros(num_steps)
    P[0] = P0
    for i in range(1, num_steps):
        P[i] = runge_kutta(h, P[i-1], Humedad[i-1], r, K0, w1, c)
    return P
from scipy.optimize import minimize


# Función de error a minimizar
def error_function(params, t, P_obs, Humedad_obs):
    K0, w1, r, c, P0 = params
    P_pred = simulate_model(r, K0, w1, c, P0, Humedad_obs, h=1, num_steps=len(t))
    error = np.sum((P_pred - P_obs)**2)
    return error

# Estimación inicial de los parámetros
#K0,w1, r, c, P0
params_init = [12, 0.01, 0.01, 1, 0.2]

# Optimización para encontrar los mejores parámetros
result = minimize(error_function, params_init, args=(tiempo_observado, poblacion_observada, Humedad), method='Nelder-Mead')

# Parámetros óptimos
K0_opt, w1_opt, r_opt, c_opt, P0_opt = result.x

# Simulación del modelo con los parámetros óptimos
IAF_simulada = simulate_model(r_opt, K0_opt, w1_opt, c_opt, P0_opt, Humedad, h=1, num_steps=len(tiempo_observado))

# Graficar los resultados
plt.scatter(tiempo_observado, poblacion_observada, label='Datos Observados', color='blue')
plt.plot(tiempo_observado, IAF_simulada, label='Predicción del Modelo', color='red')
plt.xlabel('Tiempo')
plt.ylabel('IAF')
plt.legend()
plt.show()

print("Parámetros óptimos:")
print(f"K0_opt: {K0_opt}")
print(f"w1_opt: {w1_opt}")
print(f"r_opt: {r_opt}")
print(f"c_opt: {c_opt}")
print(f"P0_opt: {P0_opt}")
