import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
filePath = 'C:/Users/Usuario/Desktop/maizmodelo/Dataset/Ajustados2.csv'
df = pd.read_csv(filePath)
DATOS={'HUMEDAD':[25,28,35,75,100]}
df1=pd.DataFrame(DATOS)
def ajustar_y_visualizar(df, columna_datos):
    # Obtener los datos del DataFrame
    datos = df[columna_datos].dropna().values

    # Ajustar los datos a una distribución normal
    media, desviacion_estandar = norm.fit(datos)

    # Crear un histograma de los datos
    plt.hist(datos, bins=122, density=True, alpha=0.6, color='g')

    # Crear una línea de ajuste basada en la distribución normal
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    p = norm.pdf(x, media, desviacion_estandar)
    plt.plot(x, p, 'k', linewidth=2)

    # Agregar información al gráfico
    title = "Fit results: media = %.2f,  desviacion = %.2f" % (media, desviacion_estandar)
    plt.title(title)

    # Mostrar el gráfico
    plt.show()

# Ejemplo de uso
# Supongamos que tienes un DataFrame llamado 'df' con una columna 'datos'
# Puedes ajustar y visualizar los datos así:
ajustar_y_visualizar(df, 'DIF')