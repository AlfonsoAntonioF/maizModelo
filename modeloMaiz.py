import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sympy as sym
#Graficos
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
# configuración de matplotlib
plt.rcParams['image.cmap']="bwr"
plt.rcParams['figure.dpi']="100"
plt.rcParams['savefig.bbox']="tight"
style.use('ggplot') or plt.style.use('ggplot')
# Graficos interactivos
import plotly.express as px
import plotly.graph_objects as go
#Procesado y modelado
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import statsmodels as sms
# ajustes de datos 
from scipy.optimize import curve_fit
#from sklearn.datasets import load_boston
from statsmodels.stats.outliers_influence import variance_inflation_factor # para calcular el VIF
# configuración de warnings
import warnings
warnings.filterwarnings('ignore')

# #Importamos la base de datos 
filePath = 'C:/Users/Usuario/Desktop/maizmodelo/Dataset/Ajustados2.csv'
dataSet = pd.read_csv(filePath)
#print(dataSet)
dias = dataSet['DIAS']
preciPluv = dataSet['PRECIP']
evap = dataSet['EVAP']
tmax = dataSet['TMAX']
tmin = dataSet['TMIN']
hl = dataSet['HORAS LUZ']
HI = dataSet['HI']
IAF = dataSet['IAF']

# Funciones
def grafica_datos(x, y, x_titulo, y_titulo, titulo):
    """
    Esta funcion grafica datos de un dataframe
    
    :param x: Los valores del eje x
    :param y: Los valores del eje y
    :param x_titulo: El título del eje x
    :param y_titulo: El título del eje y
    :param titulo: El título del gráfico
    """
    datos = go.Scatter(x=x, y=y, mode='markers')
    etiquetas = go.Layout(title=titulo,
                          xaxis=dict(title=x_titulo),
                          yaxis=dict(title=y_titulo))
    fig = go.Figure(data=[datos], layout=etiquetas)
    fig.show()
    
# funcion de regresion lineal multiple por partes
def regression_by_intervals(dias, precip, evap, tmax, tmin, hl, iaf):
    """
    Esta funcion retorna los parametros de la regresion lineal por intervalo de tiempo de un dataframe
    
    :param dias: Los valores del los dias(V. predictora)
    :param precip: Los valores de la Precipitacion Pluvial(V. predictora)
    :param evap: Los valores de la precipitacion(V. predictora)
    :param tmax: Los valores de la temperatura maxima(V. predictora)
    :param tmin: Los valores de la temperatura minima(V. predictora)
    :param hl: Los valores de las horas de luz(V. predictora)
    :param iaf: los datos del Indice de Area Foliar(V a predecir)
    """
    data = pd.DataFrame({'Dias': dias, 'Precip': precip, 'Evap':evap, 'Tmax': tmax, 'Tmin':tmin, 'Hl':hl, 'IAF': iaf})
    
    results = []

    for i in range(1, len(dias)):
        interval_data = data[data['Dias'].between(dias[i-1], dias[i])]

        X = interval_data[['Precip','Evap', 'Tmax', 'Tmin', 'Hl' ]]
        X = sm.add_constant(X)
        y = interval_data['IAF']

        model = sm.OLS(y, X).fit()

        # Almacena los resultados en una lista
        results.append({
            'Intervalo': f'Dia {dias[i-1]} al Dia {dias[i]}',
            'Coeficientes': model.params,
            'R2': model.rsquared,
            'P-valor': model.f_pvalue
        })

    return results


#Grafica de los datos de precipitacion pluvial
grafica_datos(dias, preciPluv, 
              'Dias', 'Precipitacion Pluvial mm ', 
              ' Promedio de lluvias en mm de los meses'+
              ' en los años 2017-2022 ' )

#Grafica de los datos de la evaporacion
grafica_datos(dias, evap,'Dias', 'Evaporacion', 
              ' Evaporacion promedio en mm de los meses'+
              ' en los años 2017-2022 ')

#Grafica de los datos de las temperaturas maximas 
grafica_datos(dias, tmax,'Dias','Temperatura Maxima C°',
              ' Temperatura máxima en °C de los meses'+
              ' en los años 2017-2022 ')

#Grafica de los datos de la temperatura minimas 
grafica_datos(dias, tmin,'Dias','Temperatura Minima C°',
              ' Temperatura mínima en °C de los meses'+
              ' en los años 2017-2022 ')

#Grafica de los datos de HI
grafica_datos(dias, HI,'Dias','indice de desarrollo','Indice de Cosecha Ajustado')

#Grafica de los datos de biomasa
grafica_datos(dias, IAF,'Dias','Indice de Area Foliar','Gráfica del crecimiento de la planta de maíz ')

# Llamada a la función para obtener los alpha_i
results = regression_by_intervals(dias, preciPluv, evap, tmax, tmin, hl, IAF)

# Imprime los resultados obtenidos
for result in results:
    print(f"{result['Intervalo']}: Coeficientes {result['Coeficientes']}, R2 {result['R2']}, P-valor {result['P-valor']}")

# Ajuste de los datos a un modelo logistico

# Función logística (modelo a ajustar)
def modelo_logistico(t, r, K):
    return K / (1 + ((K - 1) / np.exp(r * t)))

#Ajuste del modelo a los datos
params, params_covariance = curve_fit(modelo_logistico, dias, IAF, maxfev=10000)

# Parámetros ajustados
r, K = params

# Imprimir el valor de r calculado
print(f"Valor de r={r}  y k = {K}")

# Generar datos con el modelo ajustado para visualización
tiempo_prediccion = np.linspace(0, 200, 200)
poblacion_predicha = modelo_logistico(tiempo_prediccion, r, K)

# Visualización de los datos y el modelo ajustado
plt.figure(figsize=(8, 6))
plt.scatter(dias, IAF, label="Datos reales", color='blue')
plt.plot(tiempo_prediccion, poblacion_predicha, label="Modelo logístico ajustado", color='red')
plt.xlabel("Tiempo")
plt.ylabel("Indice de Área Foliar ")
plt.legend()
plt.show()



