import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm  # Import the tqdm library for progress bar
import nltk
from nltk.stem import PorterStemmer
import re
from sklearn.naive_bayes import MultinomialNB

class TwitterModel:
    def __init__(self, csv_file, model_path=r"C:\Users\naman\OneDrive\Escritorio\3r\Projecte_ML_twit_Nana\model_NB_stemming_long_70", max_message_length=140):
        self.model_path = model_path

        # Carga les dades
        try:
            column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
            self.data = pd.read_csv(csv_file, names=column_names, header=0, delimiter=",", encoding='latin1')
            print("\nTwitter data carregada correctament.")
        except FileNotFoundError:
            print(f"\nError: Fitxer {csv_file} no trobat.")
            return
        except Exception as e:
            print(f"\nError cargant les dades: {e}")
            return

        try:
            # Filtrar missatges per longitud màxima
            self.data = self.data[self.data['text'].str.len() <= max_message_length]
            print(f"\nDades filtrades. Nombre de missatges després del filtre: {len(self.data)}")

            self.y = self.data['target']  # Target variable
            self.X = self.data['text'].values  # Només el text ens interessa

            # Preprocessa i standarditza les dades
            self.X = self.preprocess_data(self.X)

            # Separem entre train i val
            self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(self.X, self.y, test_size=0.4, random_state=42)
            print("Mida de les dades d'entrenament X: ", self.X_train.shape)

            # Separem entre val i test
            self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_temp, self.y_temp, test_size=0.5, random_state=42)
            print("Mida de les dades de validació X: ", self.X_val.shape)

            # Inicialitzem el model
            self.model = MultinomialNB()

        except KeyError as e:
            print(f"\nError: Columna no trobada - {e}")
        except Exception as e:
            print(f"\nError durant el preprocessament: {e}")

    def preprocess_data(self, X):
        try:
            stemmer = PorterStemmer()

            def clean_data(text):
                text = re.sub(r'http\S+/www.\S+', '', text)  # Eliminar enllaços
                text = re.sub(r'@\w+', '', text)  # Eliminar @
                text = re.sub(r'#', '', text)  # Eliminar #
                text = re.sub(r'[^\w\s]', '', text)  # Eliminar puntuació
                text = text.lower()
                return text

            def stem_text(text):
                return ' '.join([stemmer.stem(word) for word in text.split()])

            # Aplicar neteja i stemming
            X = [clean_data(doc) for doc in X]
            X = [stem_text(doc) for doc in X]

            # Vectorització amb TF-IDF
            tfv = TfidfVectorizer(min_df=5, max_df=0.9, max_features=300000, strip_accents='unicode', lowercase=True,
                                  analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1, 2),
                                  use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words="english")
            X = tfv.fit_transform(X)

            print(f'Dimensions de X processat amb TF-IDF: {X.shape}')
            return X
        except Exception as e:
            print(f"Error processant les dades amb TF-IDF: {e}")
            return X

    def train_model(self, optimize_model=False):
        try:
            if optimize_model:
                param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0], 'fit_prior': [True, False]}
                grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, 
                                           cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2)
                grid_search.fit(self.X_train, self.y_train)
                self.model = grid_search.best_estimator_
            else:
                self.model.fit(self.X_train, self.y_train)

            joblib.dump(self.model, self.model_path)
            print(f"\nModel guardat a {self.model_path}.")
        except Exception as e:
            print(f"\nError training the model: {e}")

    def evaluar_modelo(self):
        try:
            y_pred = self.model.predict(self.X_val)
            accuracy = accuracy_score(self.y_val, y_pred)
            f1 = f1_score(self.y_val, y_pred, average='weighted')
            conf_matrix = confusion_matrix(self.y_val, y_pred)
            print(f"\nAccuracy: {accuracy * 100:.2f}%")
            print(f"\nF1 Score: {f1 * 100:.2f}%")
            self.visualitzar_matriz_confusion(conf_matrix)
        except Exception as e:
            print(f"\nError al evaluar el modelo: {e}")

    def visualizar_matriz_confusion(self, conf_matrix):
        try:
            conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', cbar=False)
            plt.title('Matriu de Confusió (en %)')
            plt.ylabel('Etiqueta Real')
            plt.xlabel('Etiqueta Predita')
            plt.show()
        except Exception as e:
            print(f"\nError al visualitzar la matriu de confusió: {e}")

# Funció principal
def main():
    archivo_datos = r"C:\Users\naman\OneDrive\Escritorio\3r\Projecte_ML_twit_Nana\training.1600000.processed.noemoticon.csv"

    try:
        max_len = int(input("Introdueix la longitud màxima dels missatges a utilitzar: "))
        print(f"Filtrant missatges amb longitud <= {max_len} caràcters...")
    except ValueError:
        print("Longitud no vàlida. Usant el valor predeterminat de 140 caràcters.")
        max_len = 140

    twitter_model = TwitterModel(archivo_datos, max_message_length=max_len)
    twitter_model.train_model(optimize_model=True)
    twitter_model.evaluar_modelo()

if __name__ == "__main__":
    print("\nEntrenant i evaluant el model...\n")
    main()
    print("\n\nDone!!!")
