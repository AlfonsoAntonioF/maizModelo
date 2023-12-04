import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
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
IAF = dataSet['IAF2020']
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

def fit_and_plot_sigmoid(df, x_column, y_column):
    """
    Ajusta los datos de un DataFrame a la mejor curva sigmoide y permite ajustar
    los puntos de máxima y mínima curvatura mediante deslizadores.

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
    print(f'parametros{initial_params}')

    # Encuentra los puntos de máxima y mínima curvatura
    max_curvature_points, min_curvature_points = curvature_points(x_data, y_data)

    # Crea una figura de Plotly
    fig = go.Figure()

    # Añade los datos y la curva sigmoide ajustada
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name='Datos'))
    fig.add_trace(go.Scatter(x=x_data, y=sigmoid(x_data, *initial_params), mode='lines', name='Ajuste Inicial'))

    # Añade puntos de máxima y mínima curvatura a la figura
    for point in max_curvature_points:
        fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode='markers', marker=dict(color='red', size=10), name='Punto de Máxima Curvatura'))
    for point in min_curvature_points:
        fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode='markers', marker=dict(color='blue', size=10), name='Punto de Mínima Curvatura'))

    # Define deslizadores para los parámetros de la curva sigmoide y los puntos de curvatura
    # sliders = [
    #     {'steps': [
    #         {'args': [[f'a{i+1}', initial_params[i]]], 'label': f'a{i+1}', 'method': 'restyle'}
    #         for i in range(3)
    #     ],
    #     'active': 0,
    #     'yanchor': 'top',
    #     'xanchor': 'left',
    #     'currentvalue': {'font': {'size': 14}, 'prefix': 'a', 'visible': True},
    #     'visible': True},
    #     {'steps': [
    #         {'args': [[f'point{x+1}', max_curvature_points[x][1]]], 'label': f'Máx. C{x+1}', 'method': 'restyle'}
    #         for x in range(len(max_curvature_points))
    #     ],
    #     'active': 0,
    #     'yanchor': 'top',
    #     'xanchor': 'left',
    #     'currentvalue': {'font': {'size': 14}, 'prefix': 'Max C', 'visible': True},
    #     'visible': True},
    #     {'steps': [
    #         {'args': [[f'point{x+1}', min_curvature_points[x][1]]], 'label': f'Mín. C{x+1}', 'method': 'restyle'}
    #         for x in range(len(min_curvature_points))
    #     ],
    #     'active': 0,
    #     'yanchor': 'top',
    #     'xanchor': 'left',
    #     'currentvalue': {'font': {'size': 14}, 'prefix': 'Min C', 'visible': True},
    #     'visible': True},
    # ]

    # Añade los deslizadores a la figura
    # fig.update_layout(sliders=sliders)

    # # Definir funciones de actualización para los deslizadores
    # def update_params(a1, a2, a3):
    #     params = (a1, a2, a3)
    #     fig.update_traces(selector={'name': 'Ajuste Inicial'}, y=sigmoid(x_data, *params))

    # def update_max_curvature(point1, point2, point3):
    #     points = [point1, point2, point3]
    #     for i, point in enumerate(max_curvature_points):
    #         fig.update_traces(selector={'name': f'Punto de Máxima Curvatura {i+1}'}, y=[points[i]])

    # def update_min_curvature(point1, point2):
    #     points = [point1, point2]
    #     for i, point in enumerate(min_curvature_points):
    #         fig.update_traces(selector={'name': f'Punto de Mínima Curvatura {i+1}'}, y=[points[i]])

    # # Añade las funciones de actualización a los deslizadores
    # for i in range(3):
    #     sliders[0]['steps'][i]['args'][0].append(f'a{i+1}')
    #     sliders[0]['steps'][i]['args'][0].append(initial_params[i])
    #     sliders[0]['steps'][i]['args'][0].append(update_params)

    # for i in range(len(max_curvature_points)):
    #     sliders[1]['steps'][i]['args'][0].append(f'point{i+1}')
    #     sliders[1]['steps'][i]['args'][0].append(max_curvature_points[i][1])
    #     sliders[1]['steps'][i]['args'][0].append(update_max_curvature)

    # for i in range(len(min_curvature_points)):
    #     sliders[2]['steps'][i]['args'][0].append(f'point{i+1}')
    #     sliders[2]['steps'][i]['args'][0].append(min_curvature_points[i][1])
    #     sliders[2]['steps'][i]['args'][0].append(update_min_curvature)

    # Personalizar el diseño de la gráfica
    fig.update_layout(
        title='Ajuste de Datos a Curva Sigmoide con Deslizadores',
        xaxis=dict(title=x_column),
        yaxis=dict(title=y_column),
    )

    # Mostrar la gráfica
    fig.show()

# Datos de ejemplo en un DataFrame


# Llamada a la función para la gráfica interactiva con deslizadores
fit_and_plot_sigmoid(dataSet, 'DIAS', 'IAF2017003')
