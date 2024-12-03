# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:13:33 2024

@author: abell
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:30:01 2024

@author: abell
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
import joblib

class TwitterModel:
    def __init__(self, csv_file, model_path=r"C:\Users\abell\OneDrive\Documentos\3r_Carrera\ML\PROJECTE_TWITTER_ML\modelo_twitter_lr_3000.joblib"):
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
            self.X = self.data['text'].values  # Process only the text column
            # Preprocess and standardize
            self.X = self.preprocess_data(self.X)

            # Split into training and temp sets
            self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(self.X, self.y, test_size=0.4, random_state=42)

            # Further split temp into validation and test sets
            self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_temp, self.y_temp, test_size=0.5, random_state=42)

            # Initialize the Logistic Regression model
            self.model = LogisticRegression(max_iter=1000)

        except KeyError as e:
            print(f"\nError: Column not found - {e}")
        except Exception as e:
            print(f"\nError during preprocessing: {e}")

    def preprocess_data(self, X):
        """TF-IDF vectorization with optional lemmatization."""
        try:
            lemmatizer = WordNetLemmatizer()

            # Lemmatize each word in the text
            def lemmatize_text(text):
                return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

            # Apply lemmatization to each document in X
            X = [lemmatize_text(doc) for doc in X]

            tfv = TfidfVectorizer(min_df=0, max_features=3000, strip_accents='unicode', lowercase=True,
                                  analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1, 2),  # Changed to bigrams
                                  use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words="english")
            X = tfv.fit_transform(X)
            X = X.toarray()

            print(f'Dimensions of processed X with TF-IDF: {X.shape}')
            return X

        except Exception as e:
            print(f"Error processing data with TF-IDF: {e}")
            return X

    def train_model(self):
        try:
            # Perform GridSearchCV to optimize the hyperparameters for Logistic Regression
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],  # Regularization parameter
                'penalty': ['l2', 'none'],     # Regularization type
                'solver': ['lbfgs', 'liblinear']  # Solvers to try
            }

            grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)

            # Save the best model
            self.model = grid_search.best_estimator_
            joblib.dump(self.model, self.model_path)
            print(f"\nBest Logistic Regression model saved at {self.model_path}.")
            print(f"Best Parameters: {grid_search.best_params_}")

        except Exception as e:
            print(f"\nError training the model: {e}")

    def evaluar_modelo(self):
        try:
            # Evaluate on the validation set
            y_pred = self.model.predict(self.X_val)
            accuracy = accuracy_score(self.y_val, y_pred)
            f1 = f1_score(self.y_val, y_pred, average='weighted')
            conf_matrix = confusion_matrix(self.y_val, y_pred)
            print(f"\nAccuracy: {accuracy * 100:.2f}%")
            print(f"\nF1 Score: {f1 * 100:.2f}%")
            self.visualizar_matriz_confusion(conf_matrix)

            # Evaluate on the test set
            y_pred_test = self.model.predict(self.X_test)
            accuracy_test = accuracy_score(self.y_test, y_pred_test)
            f1_test = f1_score(self.y_test, y_pred_test, average='weighted')
            print(f"\nTest Accuracy: {accuracy_test * 100:.2f}%")
            print(f"\nTest F1 Score: {f1_test * 100:.2f}%")

        except Exception as e:
            print(f"\nError evaluating the model: {e}")

    def visualizar_matriz_confusion(self, conf_matrix):
        try:
            conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', cbar=False, 
                        xticklabels=range(conf_matrix.shape[0]), yticklabels=range(conf_matrix.shape[0]))
            plt.title('Confusion Matrix (in %)')
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            plt.show()
        except Exception as e:
            print(f"\nError visualizing the confusion matrix: {e}")

# Main function
def main():
    archivo_datos = r"C:\Users\abell\OneDrive\Documentos\3r_Carrera\ML\PROJECTE_TWITTER_ML\sentiment_m140.txt"
    twitter_model = TwitterModel(archivo_datos)

    twitter_model.train_model()
    twitter_model.evaluar_modelo()

# Run the main function
if __name__ == "__main__":
    print("\nTraining and evaluating model with Logistic Regression...\n")
    main()
    print("\n\nDone!!!")
