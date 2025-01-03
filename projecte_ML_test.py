# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:32:14 2024

@author: abell
"""

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# import os
# from tqdm import tqdm  # Import the tqdm library for progress bar

# class TwitterModel:
#     def __init__(self, csv_file, model_path=r"C:\Users\abell\OneDrive\Documentos\3r_Carrera\ML\PROJECTE_TWITTER_ML\modelo_twitter.joblib"):
#         self.model_path = model_path

#         # Load data and handle exceptions
#         try:
#             self.data = pd.read_csv(csv_file, header=0, delimiter=",")
#             print("\nTraffic data loaded successfully.")
#         except FileNotFoundError:
#             print(f"\nError: File {csv_file} not found. Please check the path.")
#             return
#         except pd.errors.EmptyDataError:
#             print(f"\nError: The file {csv_file} is empty.")
#             return
#         except Exception as e:
#             print(f"\nError loading data: {e}")
#             return

#         try:
#             self.y = self.data['target']  # Target variable
            
#             self.X = self.data.drop(columns=['target'])
#             # Preprocess and standardize
#             self.X = self.preprocess_data(self.X)

#             # Split into training and temp sets
#             self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(self.X, self.y, test_size=0.4, random_state=42)

#             # Further split temp into validation and test sets
#             self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_temp, self.y_temp, test_size=0.5, random_state=42)

#                         # Cargar el modelo entrenado desde el archivo
#             if os.path.exists(self.model_path):
#                 try:
#                     self.model = joblib.load(self.model_path)
#                     print("\nModelo cargado desde el archivo.")
#                 except Exception as e:
#                     print(f"Error al cargar el modelo: {e}")
#             else:
#                 print(f"Modelo no encontrado en la ruta: {self.model_path}. Se procederá a entrenar un nuevo modelo.")

#         except KeyError as e:
#             print(f"\nError: Column not found - {e}")
#         except Exception as e:
#             print(f"\nError during preprocessing: {e}")

#     def preprocess_data(self, X):
#         """Faltaria implementar mètodes de stopword amb el bag of word i tot aixo"""
#         try:
#             # Identify numeric and categorical columns
#             numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
#             categorical_features = X.select_dtypes(include=['object']).columns

#             # Standardize numeric features                
#             scaler = StandardScaler()
#             X[numeric_features] = scaler.fit_transform(X[numeric_features])

#             # Factorize categorical features
#             for col in categorical_features:
#                 X[col], _ = pd.factorize(X[col])

#             print(f"Dimensiones de X procesado: {X.shape}")  # Verify the shape of the processed set
#             return X
#         except Exception as e:
#             print(f"\nError in data standardization: {e}")
#             return X  # Return X unchanged in case of an error


#     def evaluar_modelo(self):
#         print("\nEvaluando en conjunto de prueba:")
#         y_pred = self.model.predict(self.X_test)

#         accuracy = accuracy_score(self.y_test, y_pred)
#         f1 = f1_score(self.y_test, y_pred, average='weighted')
#         conf_matrix = confusion_matrix(self.y_test, y_pred)

#         print(f"\nAccuracy: {accuracy * 100:.2f}%")
#         print(f"\nF1 Score: {f1 * 100:.2f}%")
#         print("\nMatriz de Confusión:")
#         print(conf_matrix)
        
#         plt.figure(figsize=(10, 6))
#         plt.scatter(self.y_test, y_pred)
#         plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], '--r')
#         plt.title('Predicciones vs Valores Reales')
#         plt.xlabel('Valores Reales')
#         plt.ylabel('Predicciones')
#         plt.grid()
#         plt.show()

#         # # Comparar valores reales con predicciones
#         # print("\nComparativa de valores reales y predicciones:")
#         # for real, pred in zip(self.y_test, y_pred):
#         #     print(f"Real: {real:.2f}, Predicción: {pred:.2f}")

#         # Validación cruzada
#         """Es divideix en k=5 parts i s'entrena amb k-1 parts
#                 es repeteix k vegades """
#         scores = cross_val_score(self.model, self.X, self.y, cv=5)
#         print("Puntajes de validación cruzada:", scores)


#         self.visualize_feature_importance()
#         self.visualizar_matriz_confusion(conf_matrix)
#         self.generar_curvas_roc()

#     def visualize_feature_importance(self):
#         try:
#             importances = self.model.feature_importances_ * 100
#             indices = np.argsort(importances)[::-1]
#             plt.figure(figsize=(10, 6))
#             plt.title('Importancia de las características (en %)')
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
#             plt.title('Matriz de Confusión (en %)')
#             plt.ylabel('Etiqueta Real')
#             plt.xlabel('Etiqueta Predicha')
#             plt.show()
#         except Exception as e:
#             print(f"\nError al visualizar la matriz de confusión: {e}")
            
#     def generar_curvas_roc(self):
#         try:
#             plt.figure(figsize=(10, 7))
#             # Use the y_train and y_val for training and validation sets
#             for max_depth in [2, 4, 6, 8, 10]:
#                 clf = RandomForestClassifier(max_depth=max_depth, random_state=42)
                
#                 clf.fit(self.X_train, self.y_train)
    
#                 # Get the probabilities for the positive class (assumed to be 1 here)
#                 y_prob = clf.predict_proba(self.X_val)[:, 1]
                
#                 # Calculate ROC curve and AUC
#                 fpr, tpr, _ = roc_curve(self.y_val, y_prob, pos_label=1)  # Ensure to set pos_label correctly
#                 roc_auc = auc(fpr, tpr)
    
#                 # Plot the ROC curve
#                 plt.plot(fpr, tpr, label=f'max_depth = {max_depth} (AUC = {roc_auc:.2f})')
    
#             # Draw a diagonal line for random chance
#             plt.plot([0, 1], [0, 1], 'k--', lw=2)
#             plt.xlim([0.0, 1.0])
#             plt.ylim([0.0, 1.05])
#             plt.xlabel('False Positive Rate (FPR)')
#             plt.ylabel('True Positive Rate (TPR)')
#             plt.title('ROC Curves for Different max_depth Values')
#             plt.legend(loc='lower right')
#             plt.show()
#         except Exception as e:
#             print(f"\nError generating ROC curves: {e}")

# # Función principal
# def main():
#     archivo_datos = r"C:\Users\abell\OneDrive\Documentos\3r_Carrera\ML\PROJECTE_TWITTER_ML\sentiment_m140.txt"
#     twitter_model = TwitterModel(archivo_datos)
    
#     twitter_model.evaluar_modelo()

# # Ejecutar la función principal
# if __name__ == "__main__":
#     print("\nTraining and evaluating model...\n")
#     main()
#     print("\n\nDone!!!")




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from tqdm import tqdm  # Import the tqdm library for progress bar

class TwitterModel:
    def __init__(self, csv_file, model_path=r"C:\Users\abell\OneDrive\Documentos\3r_Carrera\ML\PROJECTE_TWITTER_ML\modelo_twitter_nb_3000.joblib"):
        self.model_path = model_path

        # Load data and handle exceptions
        try:
            self.data = pd.read_csv(csv_file, header=0, delimiter=",")
            print("\nTraffic data loaded successfully.")
        except FileNotFoundError:
            print(f"\nError: File {csv_file} not found. Please check the path.")
            return
        except pd.errors.EmptyDataError:
            print(f"\nError: The file {csv_file} is empty.")
            return
        except Exception as e:
            print(f"\nError loading data: {e}")
            return

        try:
            self.y = self.data['target']  # Target variable
            
            self.X = self.data['text'].values #Aqui agafem com a X només el text, ja que és lúnic que ens importa
            # Preprocess and standardize
            self.X = self.preprocess_data(self.X)

            # Split into training and temp sets
            self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(self.X, self.y, test_size=0.4, random_state=42)

            # Further split temp into validation and test sets
            self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_temp, self.y_temp, test_size=0.5, random_state=42)

                        # Cargar el modelo entrenado desde el archivo
            if os.path.exists(self.model_path):
                try:
                    self.model = joblib.load(self.model_path)
                    print("\nModelo cargado desde el archivo.")
                except Exception as e:
                    print(f"Error al cargar el modelo: {e}")
            else:
                print(f"Modelo no encontrado en la ruta: {self.model_path}. Se procederá a entrenar un nuevo modelo.")

        except KeyError as e:
            print(f"\nError: Column not found - {e}")
        except Exception as e:
            print(f"\nError during preprocessing: {e}")

    def preprocess_data(self, X):
        """Comencem ficant max_features=1000 perque sinó tardarà molt"""
        try:
        
            tfv=TfidfVectorizer(min_df=0, max_features=3000, strip_accents='unicode',lowercase =True,
                                analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1,1),
                                use_idf=True,smooth_idf=True, sublinear_tf=True, stop_words = "english") 
            X=tfv.fit_transform(X)
            X=X.toarray()
            
            print(f'Dimensions de X processat amb TF-IDF: {X.shape}')
            return X
        
        except Exception as e:
            print("Error processant les dades amb TF-IDF")
            return X


    def evaluar_modelo(self):
        
        print("\nEvaluando en conjunto de prueba:")
        
        y_pred = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(self.y_test, y_pred)

        print(f"\nAccuracy: {accuracy * 100:.2f}%")
        print(f"\nF1 Score: {f1 * 100:.2f}%")
        print("\nMatriz de Confusión:")
        print(conf_matrix)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], '--r')
        plt.title('Predicciones vs Valores Reales')
        plt.xlabel('Valores Reales')
        plt.ylabel('Predicciones')
        plt.grid()
        plt.show()

        # Comparar valores reales con predicciones
        print("\nComparativa de valores reales y predicciones:")
        for real, pred in zip(self.y_test, y_pred):
            print(f"Real: {real:.2f}, Predicción: {pred:.2f}")

        # Validación cruzada
        
        """
        
        Es divideix en k=5 parts i s'entrena amb k-1 parts
                es repeteix k vegades. Ajuda a veure la 
                capacitat generalitzadora del model,
                per veure si el model generalitza bé
                amb dades que encara no ha vist
        """
        
        scores = cross_val_score(self.model, self.X, self.y, cv=5) #Va bé per l'overfitting
        
        print("Puntajes de validación cruzada:", scores)


        self.visualize_feature_importance()
        self.visualizar_matriz_confusion(conf_matrix)
        # self.generar_curvas_roc()

      
    def visualize_feature_importance(self):
        try:
            importances = self.model.feature_importances_ * 100  # Get feature importances
            n_features = len(importances)
    
            if n_features == 0:
                print("El modelo no tiene características para mostrar.")
                return
    
            indices = np.argsort(importances)[::-1]  # Sort features by importance
    
            # If there are more than 10 features, show only the top 10
            top_n = min(n_features, 10)
    
            plt.figure(figsize=(10, 6))
            plt.title('Importancia de las características (en %)')
            plt.bar(range(top_n), importances[indices[:top_n]], align='center')
    
            # Instead of trying to access columns, use generic indices as labels
            plt.xticks(range(top_n), [f'Feature {i+1}' for i in indices[:top_n]], rotation=90)
            plt.ylabel('Importancia (%)')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"\nError al visualizar la importancia de características: {e}")



    def visualizar_matriz_confusion(self, conf_matrix):
        try:
            conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', cbar=False, 
                        xticklabels=range(conf_matrix.shape[0]), yticklabels=range(conf_matrix.shape[0]))
            plt.title('Matriz de Confusión (en %)')
            plt.ylabel('Etiqueta Real')
            plt.xlabel('Etiqueta Predicha')
            plt.show()
        except Exception as e:
            print(f"\nError al visualizar la matriz de confusión: {e}")
            
    def generar_curvas_roc(self):
        try:
            plt.figure(figsize=(10, 7))
            # Use the y_train and y_val for training and validation sets
            for max_depth in [2, 4, 6, 8, 10]:
                clf = RandomForestClassifier(max_depth=max_depth, random_state=42)
                
                clf.fit(self.X_train, self.y_train)
    
                # Get the probabilities for the positive class (assumed to be 1 here)
                y_prob = clf.predict_proba(self.X_val)[:, 1]
                
                # Calculate ROC curve and AUC
                fpr, tpr, _ = roc_curve(self.y_val, y_prob, pos_label=1)  # Ensure to set pos_label correctly
                roc_auc = auc(fpr, tpr)
    
                # Plot the ROC curve
                plt.plot(fpr, tpr, label=f'max_depth = {max_depth} (AUC = {roc_auc:.2f})')
    
            # Draw a diagonal line for random chance
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title('ROC Curves for Different max_depth Values')
            plt.legend(loc='lower right')
            plt.show()
        except Exception as e:
            print(f"\nError generating ROC curves: {e}")

# Función principal
def main():
    archivo_datos = r"C:\Users\abell\OneDrive\Documentos\3r_Carrera\ML\PROJECTE_TWITTER_ML\sentiment_m140.txt"
    twitter_model = TwitterModel(archivo_datos)
    
    twitter_model.evaluar_modelo()

# Ejecutar la función principal
if __name__ == "__main__":
    print("\nTesting and predicting model...\n")
    main()
    print("\n\nDone!!!")

