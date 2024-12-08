# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 20:55:42 2024

@author: abell
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm  # Import the tqdm library for progress bar
import nltk
from nltk.stem import WordNetLemmatizer
import re
import emoji
from textblob import TextBlob


# class TwitterModel:
#     def __init__(self, csv_file, model_path=r"C:\Users\abell\Downloads\archive (4)\logistic_model"):
#         self.model_path = model_path

#         # Carga les dades
#         try:
#             column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
#             self.data = pd.read_csv(csv_file, names=column_names, header=0, delimiter=",", encoding='latin1')
#             print("\nTwitter data carregada correctament.")
#         except FileNotFoundError:
#             print(f"\nError: Fitxer {csv_file} no trobat.")
#             return
#         except Exception as e:
#             print(f"\nError carregant les dades: {e}")
#             return

#         try:
#             self.y = self.data['target'].replace({4: 1}).values  # Convertimos target 4 a 1 para tener clases 0 y 1
#             self.X = self.data['text'].values  # Solo utilizamos el texto como característica
#             self.X = self.preprocess_data(self.X)  # Preprocesamos el texto

#             """Dividim 60% train, 20% validation, 20% test"""
#             # # Dividimos los datos en entrenamiento y validación
#             # self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(self.X, self.y, test_size=0.4, random_state=42)
#             # print("Mida de les dades d'entrenament X: ", self.X_train.shape)

#             # # Dividimos entre validación y test
#             # self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_temp, self.y_temp, test_size=0.5, random_state=42)
#             # print("Mida de les dades de validació X: ", self.X_val.shape)


#             """Dividim 80% train, 10% validation, 10% test"""
            
#             # División inicial: 80% entrenamiento, 20% restante
#             self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

#             # División del 20% restante en 10% validación y 10% prueba
#             self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_temp, self.y_temp, test_size=0.5, random_state=42)

#             print(f"Tamaño de entrenamiento: {self.X_train.shape}, Validación: {self.X_val.shape}, Prueba: {self.X_test.shape}")
            
            
#             # Inicializamos el modelo con parámetros predefinidos
#             self.model = LogisticRegression(random_state=42, max_iter=100, solver='lbfgs', class_weight='balanced')

#         except KeyError as e:
#             print(f"\nError: Columna no trobada - {e}")
#         except Exception as e:
#             print(f"\nError durant el preprocessament: {e}")

#     def preprocess_data(self, X):
#         try:
#             lemmatizer = WordNetLemmatizer()

#             def clean_data(text):
#                 text = re.sub(r'http\S+/www.\S+', '', text)  # Eliminar URLs
#                 text = re.sub(r'@\w+', '', text)  # Eliminar menciones
#                 text = re.sub(r'#', '', text)  # Eliminar hashtags
#                 text = re.sub(r'[^\w\s]', '', text)  # Eliminar puntuación
#                 text = text.lower()
#                 return text

#             def lemmatize_text(text):
#                 return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

#             # Aplicamos limpieza y lematización
#             X = [clean_data(doc) for doc in X]
#             X = [lemmatize_text(doc) for doc in X]

#             tfv = TfidfVectorizer(min_df=5, max_df=0.9, max_features=10000, strip_accents='unicode', lowercase=True,
#                                   analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1, 2),
#                                   use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words="english")
#             X = tfv.fit_transform(X)

#             print(f'Dimensions de X processat amb TF-IDF: {X.shape}')
#             return X

#         except Exception as e:
#             print(f"Error processant les dades amb TF-IDF: {e}")
#             return X

#     def train_model(self, optimize_model=False):
#         try:
#             if optimize_model:
#                 # Definimos los parámetros para GridSearch
#                 param_grid = {
#                     'C': [0.01, 0.1, 1, 10],  # Regularización
#                     'solver': ['lbfgs', 'liblinear'],  # Métodos de optimización
#                     'penalty': ['l2'],  # Penalty para regularización
#                     'class_weight': ['balanced', None]  # Manejo de clases desbalanceadas
#                 }

#                 grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid,
#                                            cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2)

#                 grid_search.fit(self.X_train, self.y_train)

#                 print(f"Best parameters: {grid_search.best_params_}")
#                 print(f"Best f1 score: {grid_search.best_score_}")

#                 self.model = grid_search.best_estimator_

#             else:
#                 self.model.fit(self.X_train, self.y_train)

#             # Guardamos el modelo entrenado
#             joblib.dump(self.model, self.model_path)
#             print(f"\nModel guardat a {self.model_path}.")

#         except Exception as e:
#             print(f"\nError training the model: {e}")

#     def evaluar_modelo(self):
#         try:
#             y_pred = self.model.predict(self.X_val)
#             accuracy = accuracy_score(self.y_val, y_pred)
#             f1 = f1_score(self.y_val, y_pred, average='weighted')
#             conf_matrix = confusion_matrix(self.y_val, y_pred)
#             print(f"\nAccuracy: {accuracy * 100:.2f}%")
#             print(f"\nF1 Score: {f1 * 100:.2f}%")
#             self.visualizar_matriz_confusion(conf_matrix)
#         except Exception as e:
#             print(f"\nError al evaluar el modelo: {e}")

#     def visualizar_matriz_confusion(self, conf_matrix):
#         try:
#             conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
#             plt.figure(figsize=(8, 6))
#             sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', cbar=False,
#                         xticklabels=[0, 1], yticklabels=[0, 1])
#             plt.title('Matriu de Confusió (en %)')
#             plt.ylabel('Etiqueta Real')
#             plt.xlabel('Etiqueta Predita')
#             plt.show()
#         except Exception as e:
#             print(f"\nError al visualitzar la matriu de confusió: {e}")


# # Función principal
# def main():
#     archivo_datos = r"C:\Users\abell\Downloads\archive (4)\training.1600000.processed.noemoticon.csv"
#     twitter_model = TwitterModel(archivo_datos)

#     # Entrenamiento con parámetros predefinidos
#     twitter_model.train_model(optimize_model=False)

#     # Evaluación
#     twitter_model.evaluar_modelo()

#     # Optimización con GridSearchCV
#     twitter_model.train_model(optimize_model=False)

#     # Evaluación nuevamente tras optimizar
#     twitter_model.evaluar_modelo()


# # Ejecutar la función principal
# if __name__ == "__main__":
#     print("\nEntrenant i evaluant el model Logistic Regression...\n")
#     main()
#     print("\n\nDone!!!")


#--------------------------------------------------------------pipeline---------------------

# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline

# class TwitterModel:
#     def __init__(self, csv_file, model_path=r"C:\Users\abell\Downloads\archive (4)\logistic_model"):
#         self.model_path = model_path

#         # Carga les dades
#         try:
#             column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
#             self.data = pd.read_csv(csv_file, names=column_names, header=0, delimiter=",", encoding='latin1')
#             print("\nTwitter data carregada correctament.")
#         except FileNotFoundError:
#             print(f"\nError: Fitxer {csv_file} no trobat.")
#             return
#         except Exception as e:
#             print(f"\nError carregant les dades: {e}")
#             return

#         try:
#             self.y = self.data['target'].replace({4: 1}).values  # Convertimos target 4 a 1 para tener clases 0 y 1
#             self.X = self.data['text'].values  # Solo utilizamos el texto como característica
#             self.X = self.preprocess_data(self.X)  # Preprocesamos el texto

#             """Dividim 80% train, 10% validation, 10% test"""
#             # División inicial: 80% entrenamiento, 20% restante
#             self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

#             # División del 20% restante en 10% validación y 10% prueba
#             self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_temp, self.y_temp, test_size=0.5, random_state=42)

#             print(f"Tamaño de entrenamiento: {self.X_train.shape}, Validación: {self.X_val.shape}, Prueba: {self.X_test.shape}")
            
#             # Creem la pipeline
#             self.model = make_pipeline(
#                 StandardScaler(with_mean=False),  # Necessari per dades disperses (sparse matrix)
#                 LogisticRegression(random_state=42, max_iter=500, solver='saga', class_weight='balanced')
#             )

#         except KeyError as e:
#             print(f"\nError: Columna no trobada - {e}")
#         except Exception as e:
#             print(f"\nError durant el preprocessament: {e}")

#     def preprocess_data(self, X):
#         try:
#             lemmatizer = WordNetLemmatizer()

#             def clean_data(text):
#                 text = re.sub(r'http\S+/www.\S+', '', text)  # Eliminar URLs
#                 text = re.sub(r'@\w+', '', text)  # Eliminar menciones
#                 text = re.sub(r'#', '', text)  # Eliminar hashtags
#                 text = re.sub(r'[^\w\s]', '', text)  # Eliminar puntuación
#                 text = text.lower()
#                 # text = emoji.replace_emoji(text, replace='')  # Elimina emojis
#                 # text = str(TextBlob(text).correct())  # Corregeix ortografia
#                 return text

#             def lemmatize_text(text):
#                 return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

#             # Aplicamos limpieza y lematización
#             X = [clean_data(doc) for doc in X]
#             X = [lemmatize_text(doc) for doc in X]

#             tfv = TfidfVectorizer(min_df=5, max_df=0.9, max_features=10000, strip_accents='unicode', lowercase=True,
#                                   analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1, 2),
#                                   use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words="english")
#             X = tfv.fit_transform(X)

#             print(f'Dimensions de X processat amb TF-IDF: {X.shape}')
#             return X

#         except Exception as e:
#             print(f"Error processant les dades amb TF-IDF: {e}")
#             return X

#     def train_model(self, optimize_model=False):
#         try:
#             self.model.fit(self.X_train, self.y_train)
#             # Guardamos el modelo entrenado
#             joblib.dump(self.model, self.model_path)
#             print(f"\nModel guardat a {self.model_path}.")
#         except Exception as e:
#             print(f"\nError training the model: {e}")
            
            
# # Función principal
# def main():
#     archivo_datos = r"C:\Users\abell\Downloads\archive (4)\training.1600000.processed.noemoticon.csv"
#     twitter_model = TwitterModel(archivo_datos)

#     # Entrenamiento con parámetros predefinidos
#     twitter_model.train_model(optimize_model=False)

#     # Evaluación
#     twitter_model.evaluar_modelo()


# # Ejecutar la función principal
# if __name__ == "__main__":
#     print("\nEntrenant i evaluant el model Logistic Regression...\n")
#     main()
#     print("\n\nDone!!!")














#------------------------------AMB PIPELINE COMBINAT---------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import WordNetLemmatizer


class TwitterModel:
    def __init__(self, csv_file, model_path=r"C:\Users\abell\Downloads\archive (4)\logistic_model"):
        self.model_path = model_path

        # Càrrega de les dades
        try:
            column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
            self.data = pd.read_csv(csv_file, names=column_names, header=0, delimiter=",", encoding='latin1')
            print("\nDades de Twitter carregades correctament.")
        except FileNotFoundError:
            print(f"\nError: Fitxer {csv_file} no trobat.")
            return
        except Exception as e:
            print(f"\nError carregant les dades: {e}")
            return

        try:
            self.y = self.data['target'].replace({4: 1}).values  # Convertim 4 a 1 per tenir 0 i 1 com a etiquetes
            self.X = self.data['text'].values  # Utilitzem només el text

            # Preprocessem el text
            self.X = self.preprocess_data(self.X)
            
            print("Dades preprocessades --> Netejades i Lemmatitzades")

            # Dividim les dades en entrenament, validació i test
            self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_temp, self.y_temp, test_size=0.5, random_state=42)


            # Inicialitzem el pipeline
            self.pipeline = Pipeline([
                ('scaler', StandardScaler(with_mean=False)),
                ('model', LogisticRegression(random_state=42, max_iter=500, solver='lbfgs', class_weight='balanced'))
            ])

        except Exception as e:
            print(f"\nError durant la inicialització: {e}")

    def preprocess_data(self, X):
        try:
            lemmatizer = WordNetLemmatizer()

            def clean_text(text):
                text = re.sub(r'http\S+|www.\S+', '', text)  # Eliminar URLs
                text = re.sub(r'@\w+', '', text)  # Eliminar mencions
                text = re.sub(r'#', '', text)  # Eliminar hashtags
                text = re.sub(r'[^\w\s]', '', text)  # Eliminar puntuació
                text = text.lower()  # Convertir a minúscules
                return text

            def lemmatize_text(text):
                return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

            # Neteja i lematització
            X = [clean_text(doc) for doc in X]
            X = [lemmatize_text(doc) for doc in X]

            # Vectorització TF-IDF
            tfv = TfidfVectorizer(
                min_df=5, max_df=0.9, max_features=20000, strip_accents='unicode', lowercase=True,
                analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1, 2),
                use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words="english"
            )
            X = tfv.fit_transform(X)

            print(f'Dimensions de X processat amb TF-IDF: {X.shape}')
            return X
        except Exception as e:
            print(f"Error en el preprocessament: {e}")
            return X

    def train_model(self, optimize_model=False):
        try:
            if optimize_model:
                # Definim paràmetres per a GridSearch
                param_grid = {
                    'model__C': [0.01, 0.1, 1, 10],
                    'model__solver': ['lbfgs', 'liblinear'],
                    'model__class_weight': ['balanced', None]
                }

                grid_search = GridSearchCV(self.pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2)
                grid_search.fit(self.X_train, self.y_train)

                print(f"\nMillors paràmetres: {grid_search.best_params_}")
                print(f"\nMillor F1 Score: {grid_search.best_score_:.4f}")

                self.pipeline = grid_search.best_estimator_

            else:
                self.pipeline.fit(self.X_train, self.y_train)

            # Guardem el model
            joblib.dump(self.pipeline, self.model_path)
            print(f"\nModel guardat a {self.model_path}.")

        except Exception as e:
            print(f"\nError entrenant el model: {e}")

    def evaluate_model(self):
        try:
            y_pred = self.pipeline.predict(self.X_val)
            accuracy = accuracy_score(self.y_val, y_pred)
            f1 = f1_score(self.y_val, y_pred, average='weighted')
            conf_matrix = confusion_matrix(self.y_val, y_pred)

            print(f"\nAccuracy: {accuracy * 100:.2f}%")
            print(f"F1 Score: {f1 * 100:.2f}%")

            self.plot_confusion_matrix(conf_matrix)

        except Exception as e:
            print(f"\nError avaluant el model: {e}")

    def plot_confusion_matrix(self, conf_matrix):
        try:
            conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', cbar=False,
                        xticklabels=[0, 1], yticklabels=[0, 1])
            plt.title('Matriu de Confusió (en %)')
            plt.ylabel('Etiqueta Real')
            plt.xlabel('Etiqueta Predita')
            plt.show()
        except Exception as e:
            print(f"\nError visualitzant la matriu de confusió: {e}")


# Funció principal
def main():
    archivo_datos = r"C:\Users\abell\Downloads\archive (4)\training.1600000.processed.noemoticon.csv"
    twitter_model = TwitterModel(archivo_datos)

    # Entrenament amb paràmetres predefinits
    twitter_model.train_model(optimize_model=False)

    # Avaluació
    twitter_model.evaluate_model()

    # Entrenament amb optimització
    twitter_model.train_model(optimize_model=True)

    # Nova avaluació
    twitter_model.evaluate_model()


if __name__ == "__main__":
    print("\nEntrenant i avaluant el model Logistic Regression...\n")
    main()
    print("\n\nProcess completat!")

