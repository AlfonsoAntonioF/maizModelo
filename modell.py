import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import statsmodels as sms
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

# Datos observados (puedes reemplazar estos datos con tus propios datos)
#tiempo_observado = np.array([0, 10, 20, 30, 40, 50, 60])
#poblacion_observada = np.array([100, 150, 200, 250, 300, 350, 400])

# Función de crecimiento logístico modificada con capacidad de carga variable
def logistic_growth(P, t, Precipitacion, Evaporacion, Tmax, Tmin, r, K0, w1, w2, w3):
    K = K0 + w1 * Precipitacion - w2 * Evaporacion + w3 * (Tmax - Tmin)
    dPdt = r * P * (1 - P / K)
    return dPdt

# Función de error a minimizar
def error_function(params, t, P_obs, Precipitacion, Evaporacion, Tmax, Tmin):
    r, K0, w1, w2, w3, P0 = params
    P_pred = odeint(logistic_growth, P0, t, args=(Precipitacion, Evaporacion, Tmax, Tmin, r, K0, w1, w2, w3))
    error = np.sum((P_pred.flatten() - P_obs) ** 2)
    return error

# Estimación inicial de los parámetros
params_init = [0.1, 1000, 0.1, 0.05, 0.02, 100]

# Optimización para encontrar los mejores parámetros
result = minimize(error_function, params_init, args=(tiempo_observado, poblacion_observada, Precipitacion, Evaporacion, Tmax, Tmin), method='TNC')

# Parámetros óptimos
r_opt, K0_opt, w1_opt, w2_opt, w3_opt, P0_opt = result.x

# Grafica los resultados observados y predichos
tiempo_pred = np.linspace(min(tiempo_observado), max(tiempo_observado), 100)
P_pred_opt = odeint(logistic_growth, P0_opt, tiempo_pred, args=(Precipitacion, Evaporacion, Tmax, Tmin, r_opt, K0_opt, w1_opt, w2_opt, w3_opt))

plt.scatter(tiempo_observado, poblacion_observada, label='Datos Observados', color='blue')
plt.plot(tiempo_pred, P_pred_opt, label='Predicción del Modelo', color='red')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.legend()
plt.show()

print("Parámetros óptimos:")
print(f"r_opt: {r_opt}")
print(f"K0_opt: {K0_opt}")
print(f"w1_opt: {w1_opt}")
print(f"w2_opt: {w2_opt}")
print(f"w3_opt: {w3_opt}")
print(f"P0_opt: {P0_opt}")
