"""
Created on Sat Nov 16 18:30:01 2024

@author: abell
"""

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_extraction.text import TfidfVectorizer
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# from tqdm import tqdm  # Import the tqdm library for progress bar
# import nltk
# from nltk.stem import WordNetLemmatizer
# import re
# from sklearn.naive_bayes import MultinomialNB
# import time


# class TwitterModel:
#     def __init__(self, csv_file, model_path=r"C:\Users\abell\Downloads\archive (4)\model_NB_250968"):
#         self.model_path = model_path

#         # Carga les dades
#         try:
#             column_names=['target','ids','date','flag','user','text']
#             self.data = pd.read_csv(csv_file, names=column_names, header=0, delimiter=",",encoding='latin1') #Si no funciona l'encoding afegir encoding= 'latin1'
#             print("\nTwitter data carregada correctament.")
            
#         except FileNotFoundError:
#             print(f"\nError: Fitxer {csv_file} no trobat.")
#             return
        
#         except Exception as e:
#             print(f"\nError cargant les dades: {e}")
#             return

#         try:
#             self.y = self.data['target'] # Target variable
            
#             self.X = self.data['text'].values #Aqui agafem com a X només el text, ja que és lúnic que ens importa
            
#             # Preprocessa i standarditza les dades
#             self.X = self.preprocess_data(self.X)
            
#             print("Dades netejades i Lemmatitzades")

#             # Separem entre train i val
#             self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

#             print("Mida de les dades d'entrenament X: ", self.X_train.shape)
            
#             # Separem entre val i test
            
#             self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_temp, self.y_temp, test_size=0.5, random_state=42)
#             print("Mida de les dades de validació X: ", self.X_val.shape)


#             # Inicialitzem el model
            
#             """Utilitzem aquest model més simple per fer el 
#                 GridsearchCV sinó utilitzem el d'abaix"""
                
                
#             self.model=MultinomialNB(alpha=2.0, fit_prior=True)

#         except KeyError as e:
#             print(f"\nError: Columna no trobada - {e}")
#         except Exception as e:
#             print(f"\nError durant el preprocessament: {e}")



#     def preprocess_data(self,X):
#         """Comencem ficant max_features=1000 perque sinó tardarà molt"""
#         try:
#             # Inicializar el lematizador de WordNet
#             lemmatizer = WordNetLemmatizer()
            
#             #Funció per netejar el csv
#             def clean_data(text):
#                 text=re.sub(r'http\S+/www.\S+', '',text) #Eliminar els enllaços
#                 text=re.sub(r'@\w+', '',text) #Eliminar els @
#                 text=re.sub(r'#', '',text) #Eliminar els #
#                 text=re.sub(r'[^\w\s]', '',text) #Eliminar les puntuacions
#                 text = text.lower()
#                 return text
    
#             # Función per lemmatitzar el codi
#             def lemmatize_text(text):
                
#                 """
                
#                     Lemmatitzem el text (stemming), per tal de portar
#                      les paraules a la seva arrel, aixi pot fer
#                      un millor entrenament
                     
#                 """
                
#                 return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
            
            
    
#             # Aplicam la neteja del nostre dataset
#             X = [clean_data(doc) for doc in X]
        
#             # Apliquem la lematització a cada documento a X
            
#             start_time=time.time()
#             X = [lemmatize_text(doc) for doc in X]
            
#             end_time=time.time()
#             total=end_time-start_time
            
#             print(f'El temps total en segons que és tarden a lemmatitzar les dades són: {total}')
    
#             tfv = TfidfVectorizer(min_df=1, max_df=0.95, max_features=500000, strip_accents='unicode', lowercase=True,
#                                   analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1, 2),
#                                   use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words="english") #Si dona error el min_df, llavors min_df = 1, #Anem altenran (1,2) i (1,3)
#             X = tfv.fit_transform(X)
            
#             # X = X.toarray() No convertir a dense array perque sino ocupa molt, s'hauria de convertir a sparse_matrix
            
    
#             print(f'Dimensions de X processat amb TF-IDF: {X.shape}')
#             return X
    
#         except Exception as e:
#             print(f"Error processant les dades amb TF-IDF: {e}")
#             return X
        

#     def train_model(self, optimize_model=False):
#         try:
#             #Si es true el paràmetre optimize_model llavors es fara el GridSearchCV
#             if optimize_model:
#                 # Define hyperparameter grid
#                 param_grid = {
#                     'alpha': [0.1, 0.5, 1.0, 2.0],  # Smoothing parameter
#                     # 'fit_prior': [True, False]  # Whether to learn class priors
#                 }
    
#                 # Creem el objecte GridSearCV 
#                 grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, 
#                                            cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    
#                 # Train el model utilitzant GridSearchCV
#                 for i in tqdm(range(10), desc="Training Model", unit="epoch"):
#                     grid_search.fit(self.X_train, self.y_train)
    

#                 print(f"Best parameters: {grid_search.best_params_}")
#                 print(f"Best f1 score: {grid_search.best_score_}")
    

#                 self.model = grid_search.best_estimator_
    
#             else:
#                 for i in tqdm(range(10), desc="Training Model", unit="epoch"):
#                     self.model.fit(self.X_train, self.y_train)
    
#             # Guardem el model que hem fet servir per poder no haver de tornar a entrenar després
#             joblib.dump(self.model, self.model_path)
#             print(f"\nModel guardat a {self.model_path}.")
    

#             # self.visualize_feature_importance()
    
#         except Exception as e:
#             print(f"\nError training the model: {e}")


#     # def optimize_model(self):
#     #     # Define the parameter space to search
#     #     param_grid = {
#     #         'n_estimators': [50, 100, 200],
#     #         'max_depth': [None, 10, 20, 30],
#     #         'min_samples_split': [2, 5, 10],
#     #         'min_samples_leaf': [1, 2, 4]
#     #     }

#     #     # Create the GridSearchCV object
#     #     grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, 
#     #                                cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2) #Fiquem f1_weighted perque les classes estan desbalançades, sino farem accuracy

#     #     # Train the model using GridSearchCV
#     #     grid_search.fit(self.X_train, self.y_train)

#     #     # Print the best parameters found
#     #     print(f"Best parameters: {grid_search.best_params_}")
#     #     print(f"Best f1 score: {grid_search.best_score_}")

#     #     # Update the model with the best parameters
#     #     self.model = grid_search.best_estimator_



#     def evaluar_modelo(self):
#         try:
#             # Evaluar el model per veure com ha treballat
#             y_pred = self.model.predict(self.X_val)
#             accuracy = accuracy_score(self.y_val, y_pred)
#             f1 = f1_score(self.y_val, y_pred, average='weighted')
#             conf_matrix = confusion_matrix(self.y_val, y_pred)
#             print(f"\nAccuracy: {accuracy * 100:.2f}%")
#             print(f"\nF1 Score: {f1 * 100:.2f}%")
#             self.visualizar_matriz_confusion(conf_matrix)

#             # Genera la ROC curve
#             # self.generar_curvas_roc()

#         except Exception as e:
#             print(f"\nError al evaluar el modelo: {e}")

#     def visualize_feature_importance(self):
#         try:
#             importances = self.model.feature_importances_ * 100
#             indices = np.argsort(importances)[::-1]
#             plt.figure(figsize=(10, 6))
#             plt.title('Importancia de las característicques (en %)')
#             plt.bar(range(len(importances)), importances[indices], align='center')
#             plt.xticks(range(len(importances)), np.array(self.data.drop(columns=['target']).columns)[indices], rotation=90)
#             plt.ylabel('Importancia (%)')
#             plt.tight_layout()
#             plt.show()
#         except Exception as e:
#             print(f"\nError al visualizar la importancia de características: {e}")

#     def visualizar_matriz_confusion(self, conf_matrix):
#         try:
#             conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
#             plt.figure(figsize=(8, 6))
#             sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', cbar=False, 
#                         xticklabels=range(conf_matrix.shape[0]), yticklabels=range(conf_matrix.shape[0]))
#             plt.title('Matriu de Confusió (en %)')
#             plt.ylabel('Etiqueta Real')
#             plt.xlabel('Etiqueta Predita')
#             plt.show()
            
#         except Exception as e:
#             print(f"\nError al visualitzar la matriu de confusió: {e}")
            
#     
# def generar_curvas_roc(self, X, y, title='ROC Curve'):
#     try:
#         y_prob = self.model.predict_proba(X)[:, 1]
#         fpr, tpr, _ = roc_curve(y, y_prob)
#         roc_auc = auc(fpr, tpr)
#         plt.figure(figsize=(8, 6))
#         plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
#         plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title(title)
#         plt.legend(loc="lower right")
#         plt.grid(True)
#         plt.show()
#     except Exception as e:
#         print(f"Error generating ROC curve: {e}")
# (self):
#         try:
#             plt.figure(figsize=(10, 7))
            
#             # Utilitzem el  y_train and y_val pel training i validation conjunts
#             for max_depth in [2, 4, 6, 8, 10]:
#                 clf = RandomForestClassifier(max_depth=max_depth, random_state=42)
#                 clf.fit(self.X_train, self.y_train)
    
#                 #Agafem les probabilitats per la classe postiva (Asumim que es 1 aqui)
#                 y_prob = clf.predict_proba(self.X_val)[:, 1]
                
#                 # Calcula ROC curve i AUC
#                 fpr, tpr, _ = roc_curve(self.y_val, y_prob, pos_label=1)  # Assegurem que el pos_label és correcte
#                 roc_auc = auc(fpr, tpr)
    
#                 # Plottejem la ROC curve
#                 plt.plot(fpr, tpr, label=f'max_depth = {max_depth} (AUC = {roc_auc:.2f})')
    
#             # Dibuixem una  diagonal per les possiblitats random
#             plt.plot([0, 1], [0, 1], 'k--', lw=2)
#             plt.xlim([0.0, 1.0])
#             plt.ylim([0.0, 1.05])
#             plt.xlabel('False Positive Rate (FPR)')
#             plt.ylabel('True Positive Rate (TPR)')
#             plt.title('ROC Curves for Different max_depth Values')
#             plt.legend(loc='lower right')
#             plt.show()
            
#         except Exception as e:
#             print(f"\nError generant ROC curves: {e}")

# # Función principal
# def main():
#     archivo_datos = r"C:\Users\abell\Downloads\archive (4)\training.1600000.processed.noemoticon.csv"
#     twitter_model = TwitterModel(archivo_datos)

#     twitter_model.train_model(False)
    
#     twitter_model.evaluar_modelo()

# # Ejecutar la función principal
# if __name__ == "__main__":
#     print("\nEntrenant i evaluant el  model...\n")
#     main()
#     print("\n\nDone!!!")



#------------APARTIR D'AQUI CODI QUE FUNCIONA AMB MAIN-------------
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class NaiveBayesModel:
    def __init__(self, X_train, y_train, vectorizer, model_path="model_NB.joblib"):
        """Inicialitza el model Naive Bayes."""
        self.model_path = model_path
        self.X_train = X_train
        self.y_train = self.convert_labels(y_train)  # Convertir etiquetes
        self.vectorizer = vectorizer
        self.model = MultinomialNB(alpha=1.0)

    def convert_labels(self, y):
        """Converteix les etiquetes 4 a 1 per a un format binari."""
        print("Converting labels: replacing 4 with 1...")
        return [1 if label == 4 else label for label in y]

    def train_model(self, X_val=None, y_val=None, optimize_model=False):
        """Entrena el model i avalua al moment."""
        try:
            if optimize_model:
                from sklearn.model_selection import GridSearchCV
                param_grid = {"alpha": [0.1, 0.5, 1.0, 2.0]}
                grid_search = GridSearchCV(
                    estimator=self.model,
                    param_grid=param_grid,
                    cv=5,
                    scoring="accuracy",
                    n_jobs=-1,
                    verbose=2,
                )
                grid_search.fit(self.X_train, self.y_train)
                self.model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            else:
                self.model.fit(self.X_train, self.y_train)

            # Guardar el model
            joblib.dump(self.model, self.model_path)
            print(f"\nModel guardat en {self.model_path}.")

            # Avalua el model si es proporcionen dades de validació
            if X_val is not None and y_val is not None:
                self.evaluar_modelo(X_val, y_val)
                self.generar_precision_recall_curve(X_val, y_val)

        except Exception as e:
            print(f"\nError entrenant el model: {e}")

    def evaluar_modelo(self, X_val, y_val):
        """Avalua el model amb Accuracy, F1-Score i Matriu de Confusió."""
        try:
            y_val_binary = self.convert_labels(y_val)
            y_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val_binary, y_pred)
            f1 = f1_score(y_val_binary, y_pred, average="weighted")
            conf_matrix = confusion_matrix(y_val_binary, y_pred)

            print(f"\nAccuracy: {accuracy * 100:.2f}%")
            print(f"F1 Score: {f1 * 100:.2f}%")

            self.visualizar_matriz_confusion(conf_matrix)
        except Exception as e:
            print(f"\nError al avalua el model: {e}")

    def generar_precision_recall_curve(self, X_val, y_val):
        """Genera la Precision-Recall Curve."""
        try:
            y_val_binary = self.convert_labels(y_val)  # Convertir etiquetes
            y_prob = self.model.predict_proba(X_val)[:, 1]

            precision, recall, _ = precision_recall_curve(y_val_binary, y_prob, pos_label=1)
            average_precision = average_precision_score(y_val_binary, y_prob)

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color="purple", lw=2, label=f"AP = {average_precision:.2f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.legend(loc="lower left")
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(f"\nError generant la Precision-Recall Curve: {e}")

    def mostrar_feature_importance(self):
        """Mostra la importància de les característiques basada en feature_log_prob_."""
        try:
            feature_names = self.vectorizer.get_feature_names_out()
            log_prob = self.model.feature_log_prob_[1]  # Classe positiva

            importance = pd.DataFrame({
                'Feature': feature_names,
                'Log Probability': log_prob
            }).sort_values(by='Log Probability', ascending=False)

            print("\nTop 20 Característiques Més Importants:")
            print(importance.head(20))

            plt.figure(figsize=(10, 6))
            sns.barplot(x='Log Probability', y='Feature', data=importance.head(20))
            plt.title('Top 20 Característiques Més Importants (Naive Bayes)')
            plt.xlabel('Log Probabilitat')
            plt.ylabel('Característica')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"\nError mostrant la importància de les característiques: {e}")

    def visualizar_matriz_confusion(self, conf_matrix):
        """Visualitza la matriu de confusió en percentatge."""
        try:
            conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                conf_matrix_percentage, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"]
            )
            plt.title("Matriu de Confusió (en %)")
            plt.ylabel("Etiqueta Real")
            plt.xlabel("Etiqueta Predita")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"\nError visualitzant la matriu de confusió: {e}")
