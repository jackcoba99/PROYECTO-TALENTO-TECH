import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import streamlit as st

# Configurar la página de Streamlit
st.set_page_config(page_title='Dashboard de Mortalidad', layout='wide')

# Título del dashboard
st.title('🧪 Análisis Exploratorio de Mortalidad y Patología')

# Encabezado de sección
st.header("En este informe, realizamos un análisis exhaustivo de la mortalidad y patologia") 

st.write("""1: Conversión de columnas a formato numérico:

Se usó la función pd.to_numeric() con el parámetro errors='coerce' para convertir cualquier valor no numérico en NaN.

2: Eliminación de valores no numéricos:

Se eliminaron las filas que contenían NaN en las columnas clave con dropna().

3: Análisis Exploratorio de Datos (EDA):

Se agregó un resumen estadístico para ambos archivos (Patologia.csv y MORTALIDAD.csv), mostrando un análisis general en el dashboard.

4: Cálculo de promedios:

Se calcularon los promedios de hombres y mujeres para los años 2023 y 2024, y se mostraron en el dashboard.
También se calculó el promedio de la tasa de mortalidad.

5: Descomposición de la serie temporal:

Se mostró la descomposición de la tasa de mortalidad (tendencia, estacionalidad y ruido) usando gráficos de Matplotlib integrados en el dashboard.

""")

# Función para cargar los datos
@st.cache_data
def cargar_datos():
    df_patologia = pd.read_csv('./Proyecto_final/Patologia.csv', encoding='ISO-8859-1')
    df_mortalidad = pd.read_csv('./Proyecto_final/MORTALIDAD.csv', sep=';', encoding='ISO-8859-1')
    return df_patologia, df_mortalidad

# Cargar los archivos CSV
df_patologia, df_mortalidad = cargar_datos()

# Limpiar el archivo de "Patologia.csv"
df_patologia.columns = ['Causa', 'Hombre_2023', 'Mujer_2023', 'Indeterminado_2023', 'Total_2023', 
                        'Hombre_2024', 'Mujer_2024', 'Indeterminado_2024', 'Total_2024']
df_patologia = df_patologia.drop(0).reset_index(drop=True)

# Convertir las columnas numéricas (si hay texto no numérico, lo convierte en NaN)
for col in ['Hombre_2023', 'Mujer_2023', 'Indeterminado_2023', 'Total_2023',
            'Hombre_2024', 'Mujer_2024', 'Indeterminado_2024', 'Total_2024']:
    df_patologia[col] = pd.to_numeric(df_patologia[col], errors='coerce')  # 'coerce' convierte errores en NaN

# Eliminar filas con valores NaN si es necesario (opcional)
df_patologia = df_patologia.dropna(subset=['Hombre_2023', 'Mujer_2023', 'Hombre_2024', 'Mujer_2024'])

# Mostrar la tabla de patología en el dashboard
st.subheader('Datos de Patología')
st.dataframe(df_patologia)

# EDA para el archivo de Patología
st.subheader('Análisis Exploratorio de Datos: Patología')
st.write("Resumen Estadístico:")
st.write(df_patologia.describe())

# Calcular el promedio de hombres y mujeres en 2023 y 2024
promedio_hombres_2023 = np.mean(df_patologia['Hombre_2023'])
promedio_mujeres_2023 = np.mean(df_patologia['Mujer_2023'])
promedio_hombres_2024 = np.mean(df_patologia['Hombre_2024'])
promedio_mujeres_2024 = np.mean(df_patologia['Mujer_2024'])

st.write(f'Promedio de Hombres en 2023: {promedio_hombres_2023:.2f}')
st.write(f'Promedio de Mujeres en 2023: {promedio_mujeres_2023:.2f}')
st.write(f'Promedio de Hombres en 2024: {promedio_hombres_2024:.2f}')
st.write(f'Promedio de Mujeres en 2024: {promedio_mujeres_2024:.2f}')

# Limpiar el archivo de "MORTALIDAD.csv"
df_mortalidad['Año'] = pd.to_datetime(df_mortalidad['Año'], format='%Y')
df_mortalidad.set_index('Año', inplace=True)

# Convertir las tasas de mortalidad en formato numérico
df_mortalidad['Tasa bruta de mortalidad'] = df_mortalidad['Tasa bruta de mortalidad'].str.replace(',', '.').astype(float)

# Mostrar la tabla de mortalidad en el dashboard
st.subheader('Datos de Mortalidad')
st.dataframe(df_mortalidad)

# EDA para el archivo de Mortalidad
st.subheader('Análisis Exploratorio de Datos: Mortalidad')
st.write("Resumen Estadístico:")
st.write(df_mortalidad.describe())

# Crear una serie temporal de la tasa de mortalidad total por año
mortalidad_total = df_mortalidad[df_mortalidad['Sexo'] == 'Total']['Tasa bruta de mortalidad']

# Mostrar el promedio de la tasa de mortalidad en Streamlit
promedio_mortalidad = np.mean(mortalidad_total)
st.subheader(f'Promedio de la Tasa Bruta de Mortalidad: {promedio_mortalidad:.2f}')

# Descomposición de la serie temporal (tendencia, estacionalidad y residuales/ruido)
result = seasonal_decompose(mortalidad_total, model='multiplicative', period=1)

# Visualización de la descomposición
st.subheader('Descomposición de la Serie Temporal de Mortalidad')

# Crear gráficos usando Matplotlib
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
result.observed.plot(ax=ax1, title='Serie Temporal Observada')
result.trend.plot(ax=ax2, title='Tendencia')
result.seasonal.plot(ax=ax3, title='Estacionalidad')
result.resid.plot(ax=ax4, title='Ruido (Residuales)')

plt.tight_layout()

# Mostrar los gráficos en Streamlit
st.pyplot(fig)


