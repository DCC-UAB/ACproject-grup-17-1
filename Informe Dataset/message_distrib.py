# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:07:50 2024

@author: Sahel
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_text_length(input_file):
    """
    Analitza la longitud dels textos i genera estadístiques i visualitzacions.

    Args:
        input_file (str): Ruta del fitxer d'entrada amb les columnes 'target' i 'text'.

    Returns:
        None
    """
    try:
        # Carregar el dataset
        column_names = ['target', 'text']
        df = pd.read_csv(input_file, names=column_names, header=0, delimiter=",", encoding='latin1')
        print("\nDades de Twitter carregades correctament.")

        # Gestionar valors nuls o no string
        print("Comprovant i netejant valors nuls o no string...")
        df['text'] = df['text'].fillna("").astype(str)  # Omple valors nuls amb una cadena buida

        # Calcular la longitud del text
        df['len_text'] = df['text'].apply(len)

        # Estadístiques descriptives
        print("Estadístiques de la longitud del text:")
        print(df['len_text'].describe())

        # Histograma de la distribució de longitud
        plt.hist(df['len_text'], bins=30, color='green', alpha=0.7, rwidth=0.8)
        plt.title('Distribució de longitud dels missatges')
        plt.xlabel('Longitud del text')
        plt.ylabel('Freqüència')
        plt.show()

        # Gràfic de boxplot per longitud vs classe
        sns.boxplot(x='target', y='len_text', data=df)
        plt.title('Longitud del text envers la seva Classe')
        plt.xlabel('Classe')
        plt.ylabel('Longitud del text')
        plt.show()

    except Exception as e:
        print(f"Error durant l'anàlisi de la longitud del text: {e}")

# Exemple d'ús
input_file = "processed_twitter_data.csv"
analyze_text_length(input_file)
