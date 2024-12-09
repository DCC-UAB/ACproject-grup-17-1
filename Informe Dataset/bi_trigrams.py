# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 23:50:52 2024

@author: Sahel
"""

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = pd.read_csv("training.1600000.processed.noemoticon.csv", names=column_names, header=0, delimiter=",", encoding='latin1')
print("\nTwitter data carregada correctament.")

# Función para obtener n-gramas más comunes
def get_top_ngrams(corpus, n=2, top_k=10):
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')  # Excluye stopwords
    ngram_counts = vectorizer.fit_transform(corpus)
    ngram_sums = ngram_counts.sum(axis=0)
    ngram_freq = [(word, ngram_sums[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    ngram_freq = sorted(ngram_freq, key=lambda x: x[1], reverse=True)
    return ngram_freq[:top_k]

# Separar tweets positivos y negativos
positive_tweets = df[df['target'] == 4]['text']
negative_tweets = df[df['target'] == 0]['text']

# Obtener bigramas más comunes en tweets positivos
print("Bigrames comuns en tweets positius:")
positive_bigrams = get_top_ngrams(positive_tweets, n=2, top_k=10)
for bigram, freq in positive_bigrams:
    print(f"{bigram}: {freq}")

# Obtener bigramas más comunes en tweets negativos
print("\nBigramas comuns en tweets negatius:")
negative_bigrams = get_top_ngrams(negative_tweets, n=2, top_k=10)
for bigram, freq in negative_bigrams:
    print(f"{bigram}: {freq}")

# Obtener trigramas más comunes en tweets positivos
print("\nTrigramas comuns en tweets positius:")
positive_trigrams = get_top_ngrams(positive_tweets, n=3, top_k=10)
for trigram, freq in positive_trigrams:
    print(f"{trigram}: {freq}")

# Obtener trigramas más comunes en tweets negativos
print("\nTrigramas comuns en tweets negatius:")
negative_trigrams = get_top_ngrams(negative_tweets, n=3, top_k=10)
for trigram, freq in negative_trigrams:
    print(f"{trigram}: {freq}")
