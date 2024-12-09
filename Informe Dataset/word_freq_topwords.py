# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 23:25:11 2024

@author: Sahel
"""

from collections import Counter, defaultdict
import pandas as pd
import re

key_words=['good', 'out', 'not', 'like', 'no', 'love', 'work', 'cant', 'time', 'lol', 'want', 'night', 'think', 'thanks'
           , 'home', 'off', 'miss', 'need', 'morning', 'much', 'ill', 'twitter', 'can', 'time']
# Carrega el dataset (substitueix 'tu_dataset.csv' pel nom del teu fitxer)
df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin-1", header=None, names=["label", "id", "date", "query", "user", "text"])

# Inicialitza el diccionari per comptar paraules per classe
word_class_counts = defaultdict(lambda: {"total": 0, "Negative": 0, "Positive": 0})

# Processa cada fila del dataset
for index, row in df.iterrows():
    label = row["label"]  # Classe 0 o 4
    text = row["text"].lower()
    words = re.findall(r'\b\w+\b', text)

    for word in words:
        if word in key_words:
            word_class_counts[word]["total"] += 1
            if label == 0:
                word_class_counts[word]["Negative"] += 1
            elif label == 4:
                word_class_counts[word]["Positive"] += 1

for word in word_class_counts: 
    total= word_class_counts[word]['total']
    positive= word_class_counts[word]["Positive"]
    word_class_counts[word]["Positive %"]= (positive/total)*100

# Converteix el diccionari en un DataFrame per ordenar-lo i mostrar-lo millor
result = pd.DataFrame.from_dict(word_class_counts, orient="index")
result = result.sort_values(by="total", ascending=False)

# Mostra les 20 paraules més freqüents amb el nombre total i el desglossament per classes
print(result)