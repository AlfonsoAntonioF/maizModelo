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
# Graficos interactivos
import plotly.express as px
import plotly.graph_objects as go
# ===========================================================

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
#from sklearn.datasets import load_boston
from statsmodels.stats.outliers_influence import variance_inflation_factor # para calcular el VIF
# configuración de matplotlib
plt.rcParams['image.cmap']="bwr"
plt.rcParams['figure.dpi']="100"
plt.rcParams['savefig.bbox']="tight"
style.use('ggplot') or plt.style.use('ggplot')
# configuración de warnings
import warnings
warnings.filterwarnings('ignore')

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


#Importamosla base de datos 
filePath = 'C:/Users/Usuario/Desktop/maizmodelo/Dataset/DatosSiembraPromedios.csv'
dataSet = pd.read_csv(filePath)
print(dataSet)
dias = dataSet['DIAS']
preciPluv = dataSet['PRECIP']
evap = dataSet['EVAP']
tmax = dataSet['TMAX']
tmin = dataSet['TMIN']
hl = dataSet['HORAS LUZ']

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



