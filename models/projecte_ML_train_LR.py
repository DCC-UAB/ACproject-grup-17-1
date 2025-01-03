# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 20:55:42 2024

@author: abell
"""


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
        self.y_train = self.convert_labels(y_train)  
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
                    self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2
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
