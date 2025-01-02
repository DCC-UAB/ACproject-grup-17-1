# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 20:55:42 2024

@author: abell
"""

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
# from sklearn.feature_extraction.text import TfidfVectorizer
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# from tqdm import tqdm  # Import the tqdm library for progress bar
# import nltk
# from nltk.stem import WordNetLemmatizer
# import re
# import emoji
# from textblob import TextBlob


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







#-------------------LOGISTIC REGRESSION-------------------
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     accuracy_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
# )
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np

# class LogisticRegressionModel:
#     def __init__(self, X_train, y_train, model_path="model_LR.joblib"):
#         """Inicialitza el model Logistic Regression."""
#         self.model_path = model_path
#         self.X_train = X_train
#         self.y_train = y_train

#         # Inicialitzem el model Logistic Regression
#         self.model = LogisticRegression(
#             random_state=42, max_iter=500, C=0.5, solver='lbfgs'
#         )

#     def train_model(self, X_val, y_val, optimize_model=False):
#         """Entrena el model i avalua automàticament."""
#         try:
#             if optimize_model:
#                 from sklearn.model_selection import GridSearchCV
#                 param_grid = {
#                     'C': [0.01, 0.1, 1, 10],
#                     'solver': ['lbfgs', 'liblinear'],
#                     'class_weight': ['balanced', None]
#                 }

#                 grid_search = GridSearchCV(
#                     self.model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2
#                 )
#                 grid_search.fit(self.X_train, self.y_train)

#                 print(f"Best parameters: {grid_search.best_params_}")
#                 print(f"Best F1 score: {grid_search.best_score_:.4f}")
#                 self.model = grid_search.best_estimator_
#             else:
#                 self.model.fit(self.X_train, self.y_train)

#             # Guardar el model
#             joblib.dump(self.model, self.model_path)
#             print(f"Model guardat a {self.model_path}.")

#             # Avalua el model automàticament
#             self.evaluar_modelo(X_val, y_val)

#         except Exception as e:
#             print(f"Error entrenant el model: {e}")

#     def evaluar_modelo(self, X_val, y_val):
#         """Avalua el model amb Accuracy, F1 Score i altres mètriques."""
#         try:
#             y_pred = self.model.predict(X_val)
#             accuracy = accuracy_score(y_val, y_pred)
#             f1 = f1_score(y_val, y_pred, average='weighted')
#             conf_matrix = confusion_matrix(y_val, y_pred)

#             print(f"\nAccuracy: {accuracy * 100:.2f}%")
#             print(f"F1 Score: {f1 * 100:.2f}%")

#             self.visualizar_matriz_confusion(conf_matrix)
#             self.generar_roc_curve(X_val, y_val)
#             self.generar_precision_recall_curve(X_val, y_val)
#         except Exception as e:
#             print(f"\nError al evaluar el modelo: {e}")

#     def generar_roc_curve(self, X_val, y_val):
#         """Genera la corba ROC-AUC."""
#         try:
#             y_score = self.model.decision_function(X_val)
#             fpr, tpr, _ = roc_curve(y_val, y_score)
#             roc_auc = auc(fpr, tpr)

#             plt.figure(figsize=(8, 6))
#             plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
#             plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
#             plt.xlim([0.0, 1.0])
#             plt.ylim([0.0, 1.05])
#             plt.xlabel('False Positive Rate')
#             plt.ylabel('True Positive Rate')
#             plt.title('Receiver Operating Characteristic (ROC)')
#             plt.legend(loc="lower right")
#             plt.grid(True)
#             plt.show()
#         except Exception as e:
#             print(f"\nError generant la ROC-AUC curve: {e}")

#     def generar_precision_recall_curve(self, X_val, y_val):
#         """Genera la Precision-Recall Curve."""
#         try:
#             y_score = self.model.decision_function(X_val)
#             precision, recall, _ = precision_recall_curve(y_val, y_score)
#             average_precision = average_precision_score(y_val, y_score)

#             plt.figure(figsize=(8, 6))
#             plt.plot(recall, precision, color="purple", lw=2, label=f"AP = {average_precision:.2f}")
#             plt.xlabel("Recall")
#             plt.ylabel("Precision")
#             plt.title("Precision-Recall Curve")
#             plt.legend(loc="lower left")
#             plt.grid(True)
#             plt.show()
#         except Exception as e:
#             print(f"\nError generant la Precision-Recall curve: {e}")

#     def visualizar_matriz_confusion(self, conf_matrix):
#         """Visualitza la matriu de confusió en percentatge."""
#         try:
#             conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
#             plt.figure(figsize=(8, 6))
#             sns.heatmap(conf_matrix_percentage, annot=True, fmt=".2f", cmap="Blues", cbar=False,
#                         xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"])
#             plt.title("Matriu de Confusió (en %)")
#             plt.ylabel("Etiqueta Real")
#             plt.xlabel("Etiqueta Predita")
#             plt.show()
#         except Exception as e:
#             print(f"\nError visualitzant la matriu de confusió: {e}")

#     def mostrar_feature_importance(self, vectorizer):
#         """Mostra la importància de les característiques basada en els coeficients."""
#         try:
#             feature_names = vectorizer.get_feature_names_out()
#             coef = self.model.coef_.flatten()

#             importance = pd.DataFrame({'Feature': feature_names, 'Coef': coef})
#             importance = importance.reindex(importance['Coef'].abs().sort_values(ascending=False).index)

#             plt.figure(figsize=(10, 6))
#             sns.barplot(x='Coef', y='Feature', data=importance.head(20))
#             plt.title('Top 20 Característiques Més Importants')
#             plt.xlabel('Coeficient del Model')
#             plt.ylabel('Característica')
#             plt.show()
#         except Exception as e:
#             print(f"\nError mostrant la importància de features: {e}")




##aquest codi funciona les metriques####

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib

class LogisticRegressionModel:
    def __init__(self, X_train, y_train, vectorizer, model_path="model_LR.joblib"):
        """Inicialitza el model de Regressió Logística."""
        self.model_path = model_path
        self.X_train = X_train
        self.y_train = self.convert_labels(y_train)  # Convertir etiquetas
        self.vectorizer = vectorizer
        self.model = LogisticRegression(
            random_state=42, max_iter=500, C=0.5, solver='lbfgs'
        )

    def convert_labels(self, y):
        """Converteix les etiquetes 4 a 1 per a un format binari."""
        print("Converting labels: replacing 4 with 1...")
        return [1 if label == 4 else label for label in y]

    def train_model(self, X_val=None, y_val=None, optimize_model=False):
        """Entrena el model de Regressió Logística amb opcions per optimitzar."""
        try:
            if optimize_model:
                from sklearn.model_selection import GridSearchCV
                param_grid = {
                    'C': [0.01, 0.1, 1, 10],
                    'solver': ['lbfgs', 'liblinear'],
                    'class_weight': ['balanced', None]
                }
                grid_search = GridSearchCV(
                    self.model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2
                )
                grid_search.fit(self.X_train, self.y_train)
                self.model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
    
            else:
                self.model.fit(self.X_train, self.y_train)
    
            joblib.dump(self.model, self.model_path)
            print(f"Model guardat a {self.model_path}.")
    
            if X_val is not None and y_val is not None:
                self.evaluar_modelo(X_val, y_val)
    
        except Exception as e:
            print(f"Error entrenant el model: {e}")


    def evaluar_modelo(self, X_val, y_val):
        try:
            y_val_binary = self.convert_labels(y_val)  # Convertir etiquetes de validació
            y_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val_binary, y_pred)
            f1 = f1_score(y_val_binary, y_pred, average='weighted')
            conf_matrix = confusion_matrix(y_val_binary, y_pred)

            print(f"\nAccuracy: {accuracy * 100:.2f}%")
            print(f"F1 Score: {f1 * 100:.2f}%")

            self.visualizar_matriz_confusion(conf_matrix)
            self.generar_roc_curve(X_val, y_val_binary)
            self.visualize_feature_importance()

        except Exception as e:
            print(f"\nError al evaluar el modelo: {e}")

    def visualizar_matriz_confusion(self, conf_matrix):
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

    def visualize_feature_importance(self):
        """Visualitza la importància de les característiques basada en els coeficients."""
        try:
            coef = np.abs(self.model.coef_.flatten())  # Absolut dels coeficients
            feature_names = self.vectorizer.get_feature_names_out()
    
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': coef
            }).sort_values(by='Importance', ascending=False)
    
            print("\nTop 20 Característiques Més Importants:")
            print(importance_df.head(20))
    
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
            plt.title('Top 20 Característiques Més Importants (Logistic Regression)')
            plt.xlabel('Importància (|Coeficient|)')
            plt.ylabel('Característica')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"\nError mostrant la importància de les característiques: {e}")

    def generar_roc_curve(self, X_val, y_val_binary):
        try:
            plt.figure(figsize=(10, 7))

            # Obtenir les puntuacions de decisió del model
            y_score = self.model.decision_function(X_val)

            # Calcular la corba ROC i l'AUC
            fpr, tpr, _ = roc_curve(y_val_binary, y_score, pos_label=1)
            roc_auc = auc(fpr, tpr)

            # Graficar la corba ROC
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.show()

        except Exception as e:
            print(f"\nError generant ROC curves: {e}")
