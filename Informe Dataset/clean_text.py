# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 18:26:44 2024

@author: Sahel
"""

import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

# Descarregar recursos de NLTK
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

def preprocess_and_save(input_file, output_file, tfidf_features=20000):
    """
    Funció única per preprocessar dades de text (inclou eliminar stopwords) 
    i guardar el resultat en un fitxer CSV.

    Args:
        input_file (str): Ruta del fitxer d'entrada (CSV).
        output_file (str): Ruta del fitxer de sortida (CSV per al text preprocessat).
        tfidf_features (int): Nombre de característiques TF-IDF a utilitzar.

    Returns:
        None
    """
    try:
        # Carregar les dades
        print("Carregant dades...")
        column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
        data = pd.read_csv(input_file, names=column_names, header=0, encoding='latin1')
        
        # Inicialitzar el lematitzador i stopwords
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        def clean_text(text):
            text = re.sub(r'http\S+|www.\S+', '', text)  # Eliminar URLs
            text = re.sub(r'@\w+', '', text)  # Eliminar mencions
            text = re.sub(r'#', '', text)  # Eliminar hashtags
            text = re.sub(r'[^\w\s]', '', text)  # Eliminar puntuació
            text = text.lower()  # Convertir a minúscules
            return text

        def lemmatize_and_remove_stopwords(text):
            words = text.split()
            words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
            return ' '.join(words)

        # Aplicar neteja, eliminació de stopwords i lematització
        print("Netejant, eliminant stopwords i lematitzant text...")
        data['clean_text'] = data['text'].apply(clean_text).apply(lemmatize_and_remove_stopwords)

        # Vectorització TF-IDF
        print("Aplicant vectorització TF-IDF...")
        tfv = TfidfVectorizer(
            min_df=5, max_df=0.9, max_features=tfidf_features, strip_accents='unicode', lowercase=True,
            analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1, 2),
            use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words="english"
        )
        X_tfidf = tfv.fit_transform(data['clean_text'])

        # Guardar resultats preprocessats
        print("Guardant dades preprocessades...")
        data[['target', 'clean_text']].to_csv(output_file, index=False, encoding='utf-8')
        save_npz(output_file.replace('.csv', '_tfidf.npz'), X_tfidf)

        print(f"Dades preprocessades guardades a {output_file}.")
        print(f"Matriu TF-IDF guardada a {output_file.replace('.csv', '_tfidf.npz')}.")
    except Exception as e:
        print(f"Error durant el preprocessament: {e}")

# Exemple d'ús:
input_file = "training.1600000.processed.noemoticon.csv"
output_file = "processed_twitter_data.csv"
preprocess_and_save(input_file, output_file)

