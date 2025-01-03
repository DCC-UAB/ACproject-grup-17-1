"""
Created on Sat Nov 16 18:30:01 2024

@author: abell
"""

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
