# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 23:50:52 2024

@author: Sahel
"""
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Carregar el dataset
column_names = ['target', 'text']
df = pd.read_csv("processed_twitter_data.csv", names=column_names, header=0, delimiter=",", encoding='latin1')
print("\nDades de Twitter carregades correctament.")

# Gestionar valors nuls: omplir amb cadenes buides i convertir a text
df['text'] = df['text'].fillna("").astype(str)

# Funció per obtenir n-grams més comuns
def get_top_ngrams(corpus, n=2, top_k=10):
    """
    Obté els n-grams més comuns d'un corpus.

    Args:
        corpus (iterable): Llista o sèrie de textos.
        n (int): Longitud dels n-grams (2 per bigrams, 3 per trigrames, etc.).
        top_k (int): Nombre màxim d'n-grams més comuns a retornar.

    Returns:
        list: Llista de tuples amb els n-grams i les seves freqüències.
    """
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')  # Exclou stopwords
    ngram_counts = vectorizer.fit_transform(corpus)
    ngram_sums = ngram_counts.sum(axis=0)
    ngram_freq = [(word, ngram_sums[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    ngram_freq = sorted(ngram_freq, key=lambda x: x[1], reverse=True)
    return ngram_freq[:top_k]

# Separar tweets positius i negatius
positive_tweets = df[df['target'] == 4]['text']
negative_tweets = df[df['target'] == 0]['text']

# Obtenir bigrames més comuns en tweets positius
print("Bigrames més comuns en tweets positius:")
positive_bigrams = get_top_ngrams(positive_tweets, n=2, top_k=10)
for bigram, freq in positive_bigrams:
    print(f"{bigram}: {freq}")

# Obtenir bigrames més comuns en tweets negatius
print("\nBigrames més comuns en tweets negatius:")
negative_bigrams = get_top_ngrams(negative_tweets, n=2, top_k=10)
for bigram, freq in negative_bigrams:
    print(f"{bigram}: {freq}")

# Obtenir trigrames més comuns en tweets positius
print("\nTrigrames més comuns en tweets positius:")
positive_trigrams = get_top_ngrams(positive_tweets, n=3, top_k=10)
for trigram, freq in positive_trigrams:
    print(f"{trigram}: {freq}")

# Obtenir trigrames més comuns en tweets negatius
print("\nTrigrames més comuns en tweets negatius:")
negative_trigrams = get_top_ngrams(negative_tweets, n=3, top_k=10)
for trigram, freq in negative_trigrams:
    print(f"{trigram}: {freq}")
