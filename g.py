import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
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


# Función de crecimiento logístico modificada con capacidad de carga variable
def logistic(P,t, Humedad, c, r, K0, w1):
    K = K0 + w1*(Humedad)
    dPdt = r * P**c * (1 - P / K)
    return dPdt

# Función de error a minimizar
def error_function(params, t, P_obs, Humedad):
    c,r, K0, w1, P0 = params
    P_pred = odeint(logistic, P0, t, args=(Humedad,c, r, K0, w1))
    error = np.sum((P_pred.flatten() - P_obs) ** 2)
    return error

# Estimación inicial de los parámetros
params_init = [1.5,0.1,13,0.1,0.1]

# Optimización para encontrar los mejores parámetros
result = minimize(error_function, params_init, args=(tiempo_observado, poblacion_observada, Humedad), method='Nelder-Mead')

# Parámetros óptimos
c_opt,r_opt, K0_opt, w1_opt, P0_opt = result.x

# Grafica los resultados observados y predichos
tiempo_pred = np.linspace(min(tiempo_observado), max(tiempo_observado), 100)
P_pred_opt = odeint(logistic, P0_opt, tiempo_pred, args=(Humedad, c_opt, r_opt, K0_opt, w1_opt))

plt.scatter(tiempo_observado, poblacion_observada, label='Datos Observados', color='blue')
plt.plot(tiempo_pred, P_pred_opt, label='Predicción del Modelo', color='red')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.legend()
plt.show()

print("Parámetros óptimos:")
print(f"r_opt: {c_opt}")
print(f"r_opt: {r_opt}")
print(f"K0_opt: {K0_opt}")
print(f"w1_opt: {w1_opt}")
print(f"P0_opt: {P0_opt}")
