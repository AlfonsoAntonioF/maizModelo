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

def fit_sigmoid(df, x_column, y_column, initial_guess=None):
    """
    Ajusta los datos de un DataFrame a la mejor curva sigmoide.

    :param df: DataFrame que contiene los datos.
    :type df: pandas.DataFrame
    :param x_column: Nombre de la columna para el eje x.
    :type x_column: str
    :param y_column: Nombre de la columna para el eje y.
    :type y_column: str
    :param initial_guess: Suposición inicial de parámetros (opcional).
    :type initial_guess: tuple
    :return: Parámetros ajustados de la curva sigmoide.
    :rtype: tuple
    """
    x_data = df[x_column].values
    y_data = df[y_column].values

    if initial_guess is None:
        initial_guess = [max(y_data), 1, np.median(x_data)]

    params, _ = curve_fit(sigmoid, x_data, y_data, p0=initial_guess)

    return tuple(params)

def plot_sigmoid_fit(df, x_column, y_column):
    """
    Crea una gráfica interactiva que ajusta datos a una curva sigmoide mediante deslizadores.

    :param df: DataFrame que contiene los datos.
    :type df: pandas.DataFrame
    :param x_column: Nombre de la columna para el eje x.
    :type x_column: str
    :param y_column: Nombre de la columna para el eje y.
    :type y_column: str
    """
    # Ajusta los datos a la curva sigmoide
    initial_params = fit_sigmoid(df, x_column, y_column)
    
    # Crea una figura de Plotly
    fig = go.Figure()

    # Añade los datos a la figura
    fig.add_trace(go.Scatter(x=df[x_column], y=df[y_column], mode='markers', name='Datos'))

    # Añade la curva sigmoide ajustada
    x_fit = np.linspace(min(df[x_column]), max(df[x_column]), 100)
    y_fit = sigmoid(x_fit, *initial_params)
    fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Ajuste Inicial'))

    # Define deslizadores para los parámetros de la curva sigmoide
    sliders = [
        {'steps': [
            {'args': [[f'a{i+1}', initial_params[i]]], 'label': f'a{i+1}', 'method': 'restyle'}
            for i in range(3)
        ],
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {'font': {'size': 14}, 'prefix': 'a', 'visible': True},
        'visible': True}
    ]

    # Añade los deslizadores a la figura
    fig.update_layout(sliders=sliders)

    # Define una función de actualización para los deslizadores
    def update_fit(a1, a2, a3):
        params = (a1, a2, a3)
        y_fit_updated = sigmoid(x_fit, *params)
        fig.update_traces(selector={'name': 'Ajuste Inicial'}, y=y_fit_updated)

    # Añade la función de actualización a los deslizadores
    for i in range(3):
        sliders[0]['steps'][i]['args'][0].append(f'a{i+1}')
        sliders[0]['steps'][i]['args'][0].append(initial_params[i])
        sliders[0]['steps'][i]['args'][0].append(update_fit)

    # Personaliza el diseño de la gráfica
    fig.update_layout(
        title='Ajuste de Datos a Curva Sigmoide',
        xaxis=dict(title=x_column),
        yaxis=dict(title=y_column),
    )

    # Muestra la gráfica
    fig.show()

# Datos de ejemplo en un DataFrame
data = {
    'X': [1, 2, 3, 4, 5, 6],
    'Y': [0.01, 0.02, 0.05, 0.95, 0.98, 0.99],
}

df = pd.DataFrame(data)

# Llamada a la función para la gráfica interactiva con deslizadores
plot_sigmoid_fit(df, 'X', 'Y')
