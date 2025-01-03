# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 21:38:21 2024

@author: abell
"""

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
import re
import time


csv_file= r"C:\Users\abell\Downloads\archive (4)\training.1600000.processed.noemoticon.csv"

column_names=['target','ids','date','flag','user','text']
data = pd.read_csv(csv_file, names=column_names, header=0, delimiter=",",encoding='latin1') #Si no funciona l'encoding afegir encoding= 'latin1'
print("\nTwitter data carregada correctament.")
        



y = data['target'] # Target variable
        
X = data['text'].values #Aqui agafem com a X només el text, ja que és lúnic que ens importa
        


def preprocess_data(X):
        try:
            stemmer = PorterStemmer()

            def clean_data(text):
                text = re.sub(r'http\S+/www.\S+', '', text)
                text = re.sub(r'@\w+', '', text)
                text = re.sub(r'#', '', text)
                text = re.sub(r'[^\w\s]', '', text)
                text = text.lower()
                return text

            def stem_text(text):
                """Aplica el stemming a cada paraula en el text."""
                return ' '.join([stemmer.stem(word) for word in text.split()])

            X = [clean_data(doc) for doc in X]
            
            start_time=time.time()
            X = [stem_text(doc) for doc in X]
            
            end_time=time.time()
            total=end_time-start_time
            
            print(f'El temps total en segons que és tarden a stemmatitzar les dades és de: {total}')


        except Exception as e:
            print(f"Error processant les dades amb TF-IDF: {e}")
            return X
        
        
X = preprocess_data(X)
        
print("Dades netejades i Stemmatitzades")
