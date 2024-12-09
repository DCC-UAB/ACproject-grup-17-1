# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:07:50 2024

@author: Sahel
"""

import pandas as pd
import matplotlib.pyplot as plt

import chardet

# Detectar la codificación del archivo

column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = pd.read_csv("training.1600000.processed.noemoticon.csv", names=column_names, header=0, delimiter=",", encoding='latin1')
print("\nTwitter data carregada correctament.")

df['len_text'] = df['text'].apply(len)

# Estadísticas
print("Estadístiques de la longitud del text:")
print(df['len_text'].describe())

# Histograma
plt.hist(df['len_text'], bins=30, color='green', alpha=0.7, rwidth=0.8)
plt.title('Distribució de longitud dels missatges')
plt.xlabel('Longitud del text')
plt.ylabel('Freqüència')
plt.show()


import seaborn as sns

# Gráfico de correlación
sns.boxplot(x='target', y='len_text', data=df)
plt.title('Longitud del text envers la seva Clase')
plt.xlabel('Clase')
plt.ylabel('Longitud del text')
plt.show()
