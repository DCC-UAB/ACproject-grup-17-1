# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:21:31 2024

@author: Sahel
"""
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Carregar el dataset netejat
column_names = ['target', 'text']
df = pd.read_csv("processed_twitter_data.csv", names=column_names, header=0, delimiter=",", encoding='latin1')
print("\nDades de Twitter carregades correctament.")

# Gestionar valors nuls i convertir a cadenes
df['text'] = df['text'].fillna("").astype(str)

# Concatenar tots els textos i comptar paraules
print("Comptant paraules més freqüents...")
todos_los_textos = ' '.join(df['text'])
palabras_mas_comunes = Counter(todos_los_textos.split()).most_common(20)

# Mostrar les paraules més freqüents
print("\nParaules més freqüents:")
for palabra, frecuencia in palabras_mas_comunes:
    print(f"{palabra}: {frecuencia}")

# Crear un gràfic de barres per a les 20 paraules més freqüents
palabras, frecuencias = zip(*palabras_mas_comunes[:20])
plt.figure(figsize=(10, 6))
plt.bar(palabras, frecuencias, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title("Paraules més freqüents en el dataset")
plt.xlabel("Paraula")
plt.ylabel("Freqüència")
plt.tight_layout()
plt.show()


 
