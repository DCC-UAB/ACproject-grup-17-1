# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:21:31 2024

@author: Sahel
"""
import pandas as pd
from collections import Counter
import re

column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = pd.read_csv("training.1600000.processed.noemoticon.csv", names=column_names, header=0, delimiter=",", encoding='latin1')
print("\nTwitter data carregada correctament.")


# Función para limpiar texto
def limpiar_texto(texto):
    return re.sub(r'[^\w\s]', '', texto.lower())

# Concatenar todos los textos y contarlos
todos_los_textos = ' '.join(df['text'].apply(limpiar_texto))
palabras_mas_comunes = Counter(todos_los_textos.split()).most_common(100)


# Mostrar palabras más comunes
print("Paraules més freqüents:")
for palabra, frecuencia in palabras_mas_comunes:
    print(f"{palabra}: {frecuencia}")
    
 
