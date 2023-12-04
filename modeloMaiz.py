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
#from sigmoide import fit_and_plot_sigmoid
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
#HI = dataSet['HI']
#IAF = dataSet['IAF2017']

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
def regression_by_intervals(dias, precip, evap, tmax, tmin, hl, iaf,output_file):
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
    # Convierte los resultados en un DataFrame
    results_df = pd.DataFrame(results)

    # Guarda el DataFrame en un archivo Excel
    results_df.to_excel(output_file, index=False)

    return results

def plot_multiple_data(df, x_column, y_columns, plot_types, title="Datos Simulados con variaciones en los parametros ", x_axis_label="Eje X", y_axis_label="Eje Y"):
    """
    Crea un gráfico de líneas y/o marcadores para varios conjuntos de datos en un solo gráfico.

    :param df: DataFrame que contiene los datos.
    :type df: pandas.DataFrame
    :param x_column: Nombre de la columna para el eje x.
    :type x_column: str
    :param y_columns: Lista de nombres de columnas para el eje y.
    :type y_columns: list
    :param plot_types: Lista de tipos de gráfico correspondientes a cada conjunto de datos.
                       Opciones: 'lines', 'markers', 'lines+markers'.
    :type plot_types: list
    :param title: Título de la gráfica (por defecto es "Gráfico de Múltiples Datos").
    :type title: str
    :param x_axis_label: Etiqueta del eje x (por defecto es "Eje X").
    :type x_axis_label: str
    :param y_axis_label: Etiqueta del eje y (por defecto es "Eje Y").
    :type y_axis_label: str
    """
    # Crear la figura
    fig = go.Figure()

    # Añadir trazas para cada conjunto de datos
    for y_column, plot_type in zip(y_columns, plot_types):
        if plot_type == 'lines':
            trace = go.Scatter(x=df[x_column], y=df[y_column], mode='lines', name=y_column)
        elif plot_type == 'markers':
            trace = go.Scatter(x=df[x_column], y=df[y_column], mode='markers', name=y_column)
        elif plot_type == 'lines+markers':
            trace = go.Scatter(x=df[x_column], y=df[y_column], mode='lines+markers', name=y_column)
        else:
            raise ValueError(f"Tipo de gráfico no reconocido: {plot_type}")

        fig.add_trace(trace)

    # Personalizar el diseño de la gráfica
    fig.update_layout(
        title=title,
        xaxis=dict(title=x_axis_label),
        yaxis=dict(title=y_axis_label),
    )

    # Mostrar la gráfica
    fig.show()
    
#Curva sigmoide
def sigmoid(x, a, b, c):
    """
    Función sigmoide.

    :param x: Variable independiente.
    :type x: np.array
    :param a: Parámetro de amplitud.
    :type a: float
    :param b: Parámetro de pendiente.
    :type b: float
    :param c: Parámetro de desplazamiento horizontal.
    :type c: float
    :return: Valor de la función sigmoide en x.
    :rtype: np.array
    """
    return a / (1 + np.exp(-b * (x - c)))
def fit_and_plot_sigmoid(df, x_column, y_column):
    """
    Ajusta los datos de un DataFrame a la mejor curva sigmoide y muestra el valor de
    cada parámetro obtenido del ajuste.

    :param df: DataFrame que contiene los datos.
    :type df: pandas.DataFrame
    :param x_column: Nombre de la columna para el eje x.
    :type x_column: str
    :param y_column: Nombre de la columna para el eje y.
    :type y_column: str
    """
    x_data = df[x_column].values
    y_data = df[y_column].values

    # Ajusta los datos a la curva sigmoide
    initial_params, _ = curve_fit(sigmoid, x_data, y_data)

    # Encuentra los puntos de máxima y mínima curvatura
    #max_curvature_points, min_curvature_points = curvature_points(x_data, y_data)

    # Crea una figura de Plotly
    fig = go.Figure()

    # Añade los datos y la curva sigmoide ajustada
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name='Datos'))
    fig.add_trace(go.Scatter(x=x_data, y=sigmoid(x_data, *initial_params), mode='lines', name='Ajuste Inicial'))

    # Añade puntos de máxima y mínima curvatura a la figura
    # for point in max_curvature_points:
    #     fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode='markers', marker=dict(color='red', size=10), name='Punto de Máxima Curvatura'))
    # for point in min_curvature_points:
    #     fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode='markers', marker=dict(color='blue', size=10), name='Punto de Mínima Curvatura'))

    # Muestra el valor de cada parámetro en la figura
    annotation_text = f'a: {initial_params[0]:.3f}<br>b: {initial_params[1]:.3f}<br>c: {initial_params[2]:.3f}'
    fig.add_annotation(
        go.layout.Annotation(
            text=annotation_text,
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor='rgba(0, 0, 0, 0.5)',
            borderwidth=1
        )
    )

    # Personalizar el diseño de la gráfica
    equation_text = f'$f(x) = \\frac{{{initial_params[0]:.3f}}}{{1 + e^{{-{initial_params[1]:.3f}(x - {initial_params[2]:.3f})}}}}$'
    fig.update_layout(
        title={'text': f'Ajuste de Datos a Curva Sigmoide ({equation_text})', 'x': 0.5, 'xanchor': 'center'},
        xaxis=dict(title=x_column),
        yaxis=dict(title=y_column),
    )

    # Mostrar la gráfica
    fig.show()
def curvature_points(x_data, y_data, num_points=2):
    """
    Encuentra los puntos de máxima y mínima curvatura en un conjunto de datos.

    :param x_data: Datos para el eje x.
    :type x_data: np.array
    :param y_data: Datos para el eje y.
    :type y_data: np.array
    :param num_points: Número de puntos de curvatura a encontrar (por defecto es 2).
    :type num_points: int
    :return: Coordenadas de los puntos de curvatura (x, y).
    :rtype: list
    """
    deriv1 = np.gradient(y_data, x_data)
    deriv2 = np.gradient(deriv1, x_data)
    curvature = deriv2 / (1 + deriv1**2)**(3/2)

    # Encuentra los índices de los puntos de máxima y mínima curvatura
    max_curvature_indices = np.argsort(curvature)[-num_points:]
    min_curvature_indices = np.argsort(curvature)[:num_points]

    # Devuelve las coordenadas de los puntos de curvatura
    max_curvature_points = list(zip(x_data[max_curvature_indices], y_data[max_curvature_indices]))
    min_curvature_points = list(zip(x_data[min_curvature_indices], y_data[min_curvature_indices]))

    return max_curvature_points, min_curvature_points
def find_last_and_curvature(df, x_column, num_points=2):
    """
    Encuentra el último dato, las coordenadas de máxima y mínima curvatura de cada columna
    en un DataFrame y grafica los resultados en una sola gráfica con Plotly.

    :param df: DataFrame que contiene los datos.
    :type df: pandas.DataFrame
    :param x_column: Nombre de la columna para el eje x.
    :type x_column: str
    :param num_points: Número de puntos de curvatura a encontrar (por defecto es 2).
    :type num_points: int
    """
    # Inicializa la figura de Plotly
    fig = go.Figure()

    for column in df.columns[6:]:
        y_data = df[column].values
        x_data = df[x_column].values

        # Encuentra el último dato
        last_datum = (x_data[-1], y_data[-1])

        # Ajusta los datos a la curva sigmoide
        initial_params, _ = curve_fit(sigmoid, x_data, y_data)

        # Encuentra los puntos de máxima y mínima curvatura
        #max_curvature_points, min_curvature_points = curvature_points(x_data, y_data, num_points)

        # Añade los datos a la figura
        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name=column))
        #fig.add_trace(go.Scatter(x=[last_datum[0]], y=[last_datum[1]], mode='markers', marker=dict(color='green', size=10), name=f'Último Dato ({column})'))
        fig.add_trace(go.Scatter(x=x_data, y=sigmoid(x_data, *initial_params), mode='lines', name='Ajuste Inicial'))

        # for i, point in enumerate(max_curvature_points):
        #     fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode='markers', marker=dict(color='red', size=10), name=f'Máx. Curvatura {i + 1} ({column})'))

        # for i, point in enumerate(min_curvature_points):
        #     fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode='markers', marker=dict(color='blue', size=10), name=f'Mín. Curvatura {i + 1} ({column})'))

    # Personaliza el diseño de la gráfica
    fig.update_layout(
        title='GRAFICA DE LOS DATOS SIMULADOS ',
        xaxis=dict(title=x_column),
        yaxis=dict(title='Valor'),
    )

    # Muestra la gráfica
    fig.show()

#find_last_and_curvature(dataSet, 'DIAS')

# Llamada a la función con los datos del DataFrame
# plot_multiple_data(dataSet, 'DIAS', ['IAF', 'IAF2017', 'IAF2018', 'IAF2019','IAF2020',
#                                      'IAF2017001', 'IAF2017002', 'IAF2017003', 'IAF2017004', 
#                                      'IAF2017005'], 
#                    ['lines', 'markers', 'lines+markers', 'lines', 'markers', 
#                     'lines+markers', 'lines', 'markers','lines+markers', 'lines'])


# Y_para = ['IAF', 'IAF2017', 'IAF2018', 
#           'IAF2019','IAF2020','IAF2017001', 
#           'IAF2017002', 'IAF2017003', 
#           'IAF2017004', 'IAF2017005']
# # Llamada a la función para la gráfica con valores de parámetros
for column in dataSet.columns[6:]:
    fit_and_plot_sigmoid(dataSet,'DIAS', y_column=column)

# #Grafica de los datos de precipitacion pluvial
# grafica_datos(dias, preciPluv, 
#               'Dias', 'Precipitacion Pluvial mm ', 
#               ' Promedio de lluvias en mm de los meses'+
#               ' en los años 2017-2022 ' )

# #Grafica de los datos de la evaporacion
# grafica_datos(dias, evap,'Dias', 'Evaporacion', 
#               ' Evaporacion promedio en mm de los meses'+
#               ' en los años 2017-2022 ')

# #Grafica de los datos de las temperaturas maximas 
# grafica_datos(dias, tmax,'Dias','Temperatura Maxima C°',
#               ' Temperatura máxima en °C de los meses'+
#               ' en los años 2017-2022 ')

# #Grafica de los datos de la temperatura minimas 
# grafica_datos(dias, tmin,'Dias','Temperatura Minima C°',
#               ' Temperatura mínima en °C de los meses'+
#               ' en los años 2017-2022 ')

# #Grafica de los datos de HI
# grafica_datos(dias, HI,'Dias','indice de desarrollo','Indice de Cosecha Ajustado')

# #Grafica de los datos de biomasa
# grafica_datos(dias, IAF,'Dias','Indice de Area Foliar','Gráfica del crecimiento de la planta de maíz ')

# output_file = 'resultados_regresion2020.xlsx'
# # Llamada a la función para obtener los alpha_i
# results = regression_by_intervals(dias, preciPluv, evap, tmax, tmin, hl, IAF,output_file)

# # Imprime los resultados obtenidos
# for result in results:
#     print(f"{result['Intervalo']}: Coeficientes {result['Coeficientes']}, R2 {result['R2']}, P-valor {result['P-valor']}")

# # Ajuste de los datos a un modelo logistico

 # Función logística (modelo a ajustar)
# def modelo_logistico(t, r, K):
#      return K / (1 + ((K - 1) / np.exp(r * t)))

# for column in dataSet.columns[6:]:
#     IAF=dataSet[column].values
# # #Ajuste del modelo a los datos
#     params, params_covariance = curve_fit(modelo_logistico, dias, IAF, maxfev=10000)

#  # Parámetros ajustados
#     r, K = params

# # # Imprimir el valor de r calculado
#     print(f"Valor de r={r}  y k = {K}")

# # # Generar datos con el modelo ajustado para visualización
#     tiempo_prediccion = np.linspace(0, 200, 200)
#     poblacion_predicha = modelo_logistico(tiempo_prediccion, r, K)

# # Visualización de los datos y el modelo ajustado
#     plt.figure(figsize=(8, 6))
#     plt.scatter(dias, IAF, label="Datos reales", color='blue')
#     plt.plot(tiempo_prediccion, poblacion_predicha, label="Modelo logístico ajustado", color='red')
#     plt.xlabel("Tiempo")
#     plt.ylabel("Indice de Área Foliar ")
#     plt.legend()
#     plt.show()



