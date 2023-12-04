import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

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

    # Muestra el valor de cada parámetro en la figura
    annotation_text = f'a1: {initial_params[0]:.3f}<br>a2: {initial_params[1]:.3f}<br>a3: {initial_params[2]:.3f}'
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
    fig.update_layout(
        title='Ajuste de Datos a Curva Sigmoide',
        xaxis=dict(title=x_column),
        yaxis=dict(title=y_column),
    )

    # Mostrar la gráfica
    fig.show()



# Llamada a la función para la gráfica con valores de parámetros
fit_and_plot_sigmoid()
