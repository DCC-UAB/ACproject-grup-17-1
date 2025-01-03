# -*- coding: utf-8 -*-


import pandas as pd

# Ruta al archivo CSV
archivo_datos = "training.1600000.processed.noemoticon.csv"

# Cargar el conjunto de datos
column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
data = pd.read_csv(archivo_datos, names=column_names, header=0, delimiter=",", encoding='latin1')
print("\nTwitter data carregada correctament.")

# Variables target y features
y = data['target']
X = data['text'].values

# Contar los valores 0 y 4 en la columna 'target'
conteig_0 = (y == 0).sum()
conteig_4 = (y == 4).sum()

# Verificar si existen otros valores
altres_valors = y[~y.isin([0, 4])].unique()

# Imprimir los resultados
print(f"\nLa cantitat de etiquetes amb valor 0 es: {conteig_0}")
print(f"La cantitat de etiquetes amb valor 4 es: {conteig_4}")

if len(altres_valors) > 0:
    print(f"\nHi ha altres valors que no son 0 o 4 a la columna'target': {altres_valors}")
else:
    print("\nNo Hi ha altres valors que no son 0 o 4 a la columna'target'.")

    
