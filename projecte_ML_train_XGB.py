# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:41:35 2024

@author: abell
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm  # Import the tqdm library for progress bar
import nltk
from nltk.stem import WordNetLemmatizer
import xgboost as xgb
from xgboost import XGBClassifier

class TwitterModel:
    def __init__(self, csv_file, model_path=r"C:\Users\abell\OneDrive\Documentos\3r_Carrera\ML\PROJECTE_TWITTER_ML\modelo_twitter_3000_xgb.joblib"):
        self.model_path = model_path

        # Load data and handle exceptions
        try:
            self.data = pd.read_csv(csv_file, header=0, delimiter=",")
            print("\nTwitter data loaded successfully.")
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
            self.X = self.data['text'].values  # Feature (text)
            # Preprocess and standardize
            self.X = self.preprocess_data(self.X)

            # Split into training and temp sets
            self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(self.X, self.y, test_size=0.4, random_state=42)

            # Further split temp into validation and test sets
            self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_temp, self.y_temp, test_size=0.5, random_state=42)

            # Initialize XGBoost model
            self.model = XGBClassifier(random_state=42, n_estimators=200, max_depth=6, learning_rate=0.1, 
                                       min_child_weight=1, subsample=0.8, colsample_bytree=0.8, 
                                       scale_pos_weight=1, eval_metric='mlogloss')

        except KeyError as e:
            print(f"\nError: Column not found - {e}")
        except Exception as e:
            print(f"\nError during preprocessing: {e}")
    
    def preprocess_data(self, X):
        """Process text with TF-IDF and lemmatization"""
        try:
            # Initialize WordNet Lemmatizer
            lemmatizer = WordNetLemmatizer()

            # Lemmatize each word in the text
            def lemmatize_text(text):
                return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

            # Apply lemmatization
            X = [lemmatize_text(doc) for doc in X]

            tfv = TfidfVectorizer(min_df=0, max_features=3000, strip_accents='unicode', lowercase=True,
                                  analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1, 1),
                                  use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words="english")
            X = tfv.fit_transform(X)
            X = X.toarray()

            print(f'Dimensions de X processat amb TF-IDF: {X.shape}')
            return X

        except Exception as e:
            print(f"Error processant les dades amb TF-IDF: {e}")
            return X

    def train_model(self):
        try:
            # Training the model
            for i in tqdm(range(10), desc="Training Model", unit="epoch"):  # Dummy loop for progress bar demonstration
                self.model.fit(self.X_train, self.y_train)

            # Save the model
            joblib.dump(self.model, self.model_path)
            print(f"\nModel saved at {self.model_path}.")
            
            self.visualize_feature_importance()  # Visualize feature importance if needed
        except Exception as e:
            print(f"\nError training the model: {e}")

    def optimize_model(self):
        # Define the parameter space for GridSearchCV
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'min_child_weight': [1, 2, 3],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
        }

        # GridSearchCV to find the best hyperparameters
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, 
                                   cv=3, scoring='f1_weighted', n_jobs=-1, verbose=2)

        # Train the model using GridSearchCV
        grid_search.fit(self.X_train, self.y_train)

        # Print the best parameters found
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best f1 score: {grid_search.best_score_}")

        # Update the model with the best parameters
        self.model = grid_search.best_estimator_

    def evaluate_model(self):
        try:
            # Evaluate on the validation set
            y_pred = self.model.predict(self.X_val)
            accuracy = accuracy_score(self.y_val, y_pred)
            f1 = f1_score(self.y_val, y_pred, average='weighted')
            conf_matrix = confusion_matrix(self.y_val, y_pred)
            print(f"\nAccuracy: {accuracy * 100:.2f}%")
            print(f"\nF1 Score: {f1 * 100:.2f}%")
            self.visualize_confusion_matrix(conf_matrix)

            # Generate ROC curve
            self.generate_roc_curve()

        except Exception as e:
            print(f"\nError al evaluar el modelo: {e}")

    def visualize_feature_importance(self):
        try:
            importances = self.model.feature_importances_ * 100
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 6))
            plt.title('Importancia de las características (en %)')
            plt.bar(range(len(importances)), importances[indices], align='center')
            plt.xticks(range(len(importances)), np.array(self.data.drop(columns=['target']).columns)[indices], rotation=90)
            plt.ylabel('Importancia (%)')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"\nError al visualizar la importancia de características: {e}")

    def visualize_confusion_matrix(self, conf_matrix):
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

    def generate_roc_curve(self):
        try:
            plt.figure(figsize=(10, 7))
            y_prob = self.model.predict_proba(self.X_val)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_val, y_prob, pos_label=1)  # Ensure to set pos_label correctly
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.show()
        except Exception as e:
            print(f"\nError generating ROC curves: {e}")

# Función principal
def main():
    archivo_datos = r"C:\Users\abell\OneDrive\Documentos\3r_Carrera\ML\PROJECTE_TWITTER_ML\sentiment_m140.txt"
    twitter_model = TwitterModel(archivo_datos)

    twitter_model.train_model()
    
    # twitter_model.optimize_model()
    
    twitter_model.evaluate_model()

# Ejecutar la función principal
if __name__ == "__main__":
    print("\nTraining and evaluating model...\n")
    main()
    print("\n\nDone!!!")
