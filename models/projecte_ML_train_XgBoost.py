# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 01:23:05 2024

@author: naman
"""

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import pandas as pd

class XGBoostModel:
    def __init__(self, X_train, y_train, vectorizer, model_path="model_XGB.joblib"):
        """Inicialitza el model XGBoost."""
        self.model_path = model_path
        self.X_train = X_train
        self.y_train = self.convert_labels(y_train)  # Convertir etiquetas
        self.vectorizer = vectorizer
        # Inicialitzem el model XGBoost
        self.model = XGBClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            eval_metric='logloss'
        )

    def convert_labels(self, y):
        """Converteix les etiquetes 4 a 1 per a un format binari."""
        print("Converting labels: replacing 4 with 1...")
        return [1 if label == 4 else label for label in y]

    def train_model(self, X_val=None, y_val=None, optimize_model=False):
        """Entrena el model XGBoost amb opcions per optimitzar."""
        try:
            if optimize_model:
                from sklearn.model_selection import GridSearchCV
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'min_child_weight': [1, 2],
                    'subsample': [0.7, 0.8, 1.0],
                    'colsample_bytree': [0.7, 0.8, 1.0]
                }
                grid_search = GridSearchCV(
                    self.model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2
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
            self.generar_curvas_roc(X_val, y_val_binary)
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
        try:
            importance = self.model.feature_importances_
            feature_names = self.vectorizer.get_feature_names_out()

            # Crear un dataframe per ordenar les característiques per importància
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
            plt.title('Top 20 Característiques Importants')
            plt.xlabel('Importància del Model')
            plt.ylabel('Característica')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"\nError mostrant la importància de les característiques: {e}")

    def generar_curvas_roc(self, X_val, y_val_binary):
        try:
            plt.figure(figsize=(10, 7))

            # Obtenir probabilitats per a la classe positiva
            y_prob = self.model.predict_proba(X_val)[:, 1]

            # Calcular la corba ROC i l'AUC
            fpr, tpr, _ = roc_curve(y_val_binary, y_prob, pos_label=1)
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
