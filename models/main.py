# # -*- coding: utf-8 -*-
# """
# Created on Wed Dec 11 18:43:16 2024

# @author: naman
# """


# # -*- coding: utf-8 -*-
# import os
# import joblib
# import pandas as pd
# from nltk.stem import WordNetLemmatizer, PorterStemmer
# import re
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# def preprocess_and_vectorize_data(dataset_path, max_features, lemmatization=True):
#     """Preprocesa los datos, los vectoriza y guarda todo en un solo archivo."""
#     suffix = 'l' if lemmatization else 's'
#     combined_file = f"model_combined_{suffix}_{max_features}.joblib"

#     if os.path.exists(combined_file):
#         print("Cargando datos y vectorizador desde archivo combinado...")
#         return joblib.load(combined_file), combined_file

#     print("Preprocesando datos y aplicando TF-IDF...")
#     lemmatizer = WordNetLemmatizer()
#     stemmer = PorterStemmer()
#     column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
#     data = pd.read_csv(dataset_path, names=column_names, header=0, delimiter=",", encoding='latin1')
#     X, y = data['text'], data['target']

#     # Preprocesar los textos
#     def preprocess_text(text):
#         text = text.lower()
#         text = re.sub(r'http\S+|www\.\S+', '', text)
#         text = re.sub(r'@\w+', '', text)
#         text = re.sub(r'#', '', text)
#         text = re.sub(r'[^\w\s]', '', text)

#         if lemmatization:
#             return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
#         else:
#             return ' '.join([stemmer.stem(word) for word in text.split()])

#     X = [preprocess_text(text) for text in X]

#     # Aplicar TF-IDF
#     vectorizer = TfidfVectorizer(
#         min_df=5, max_df=0.9, max_features=max_features, strip_accents='unicode', lowercase=True,
#         analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1, 2),
#         use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words="english"
#     )
#     X = vectorizer.fit_transform(X)

#     # Dividir datos
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#     combined_data = {
#         "X_train": X_train, "y_train": y_train,
#         "X_val": X_val, "y_val": y_val,
#         "X_test": X_test, "y_test": y_test,
#         "vectorizer": vectorizer
#     }

#     # Guardar el archivo combinado
#     joblib.dump(combined_data, combined_file)

#     print(f"Datos y vectorizador guardados en {combined_file}")

#     return combined_data, combined_file

# def list_models_in_directory():
#     """List available models in the current directory."""
#     print("\nAvailable models:")
#     models = [f for f in os.listdir() if f.startswith('model_') and f.endswith('.joblib')]
#     if models:
#         for i, model in enumerate(models):
#             print(f"{i + 1}. {model}")
#     else:
#         print("No models found.")
#     return models

# def load_model(model_path):
#     """Load a pre-trained model."""
#     print(f"Loading model from {model_path}...")
#     return joblib.load(model_path)

# def generate_roc_curve(preprocessed_data):
#     """Generate ROC Curve for the selected model."""
#     models = list_models_in_directory()
#     model_choice = int(input("Select a model to load (by number): "))

#     if model_choice < 1 or model_choice > len(models):
#         print("Invalid choice.")
#         return

#     model_path = models[model_choice - 1]
#     model = load_model(model_path)

#     X_test = preprocessed_data["X_test"]
#     y_test = preprocessed_data["y_test"]

#     try:
#         model.generar_roc_curve(X_test, y_test)
#     except AttributeError:
#         print("The selected model does not support generating an ROC curve.")
#     except Exception as e:
#         print(f"Error generating ROC curve: {e}")

# def train_model(preprocessed_data, max_features):
#     """Train a model based on user selection."""
#     print("Choose the model to train:")
#     print("1. Naive Bayes")
#     print("2. Logistic Regression")
#     print("3. Random Forest")
#     model_choice = int(input("Select an option (1-3): "))

#     print("Do you want to perform GridSearch for hyperparameter tuning? (y/n)")
#     gridsearch = input("Your choice: ").strip().lower() == 'y'

#     if model_choice == 1:
#         from projecte_ML_train_NB import NaiveBayesModel
#         ModelClass = NaiveBayesModel
#         model_name = "NB"
#     elif model_choice == 2:
#         from projecte_ML_train_LR import LogisticRegressionModel
#         ModelClass = LogisticRegressionModel
#         model_name = "LR"
#     elif model_choice == 3:
#         from projecte_ML_train_RND_FOREST import RandomForestModel
#         ModelClass = RandomForestModel
#         model_name = "RF"
#     else:
#         print("Invalid choice.")
#         return

#     print("Training the model...")

#     X_train = preprocessed_data["X_train"]
#     y_train = preprocessed_data["y_train"]
#     X_val = preprocessed_data["X_val"]
#     y_val = preprocessed_data["y_val"]

#     model_filename = f"model_{model_name}_{max_features}.joblib"
#     model = ModelClass(X_train, y_train, model_path=model_filename)
#     model.train_model(X_val, y_val, optimize_model=gridsearch)

#     model.mostrar_feature_importance(preprocessed_data["vectorizer"])

#     print(f"Model saved as {model_filename}.")

# def test_model(preprocessed_data):
#     """Test a pre-trained model."""
#     models = list_models_in_directory()
#     model_choice = int(input("Select a model to load (by number): "))

#     if model_choice < 1 or model_choice > len(models):
#         print("Invalid choice.")
#         return

#     model_path = models[model_choice - 1]
#     model = load_model(model_path)

#     X_test = preprocessed_data["X_test"]
#     y_test = preprocessed_data["y_test"]

#     try:
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred, average='weighted')
#         conf_matrix = confusion_matrix(y_test, y_pred)

#         print(f"Accuracy: {accuracy * 100:.2f}%")
#         print(f"F1 Score: {f1 * 100:.2f}%")
#     except Exception as e:
#         print(f"Error during testing: {e}")

# dataset_path = r"C:\Users\naman\OneDrive\Escritorio\pipeline_try\pipeline\training.1600000.processed.noemoticon.csv"

# def main():
#     max_features = int(input("Enter max_features for TF-IDF vectorizer (e.g., 10000): "))
#     lemmatization = int(input("Choose preprocessing: 1. Lemmatization 2. Stemming: ")) == 1

#     preprocessed_data, combined_file = preprocess_and_vectorize_data(dataset_path, max_features, lemmatization)

#     while True:
#         print("\nChoose an option:")
#         print("1. Train a model")
#         print("2. Test a model with a dataset")
#         print("3. Generate ROC Curve")
#         print("4. Exit")

#         choice = int(input("Enter your choice (1-4): "))

#         if choice == 1:
#             train_model(preprocessed_data, max_features)
#         elif choice == 2:
#             test_model(preprocessed_data)
#         elif choice == 3:
#             generate_roc_curve(preprocessed_data)
#         elif choice == 4:
#             print("Exiting...")
#             break
#         else:
#             print("Invalid choice. Please try again.")

# if __name__ == "__main__":
#     main()



# # -- coding: utf-8 --
# import os
# import joblib
# import pandas as pd
# from nltk.stem import WordNetLemmatizer, PorterStemmer
# import re
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# import sys
# # sys.path.append(r"C:\Users\abell\Downloads\Falta_curves_roc\pipeline")


# def preprocess_and_vectorize_data(dataset_path, max_features, lemmatization=True):
#     """Preprocesa los datos, los vectoriza y guarda todo en un solo archivo."""
#     suffix = 'l' if lemmatization else 's'
#     combined_file = f"model_combined_{suffix}_{max_features}.joblib"

#     if os.path.exists(combined_file):
#         print("Cargando datos y vectorizador desde archivo combinado...")
#         return joblib.load(combined_file), combined_file

#     print("Preprocesando datos y aplicando TF-IDF...")
#     lemmatizer = WordNetLemmatizer()
#     stemmer = PorterStemmer()
#     column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
#     data = pd.read_csv(dataset_path, names=column_names, header=0, delimiter=",", encoding='latin1')
#     X, y = data['text'], data['target']


#     # Preprocesar los textos
#     def preprocess_text(text):
#         text = text.lower()
#         text = re.sub(r'http\S+|www\.\S+', '', text)
#         text = re.sub(r'@\w+', '', text)
#         text = re.sub(r'#', '', text)
#         text = re.sub(r'[^\w\s]', '', text)

#         if lemmatization:
#             return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
#         else:
#             return ' '.join([stemmer.stem(word) for word in text.split()])

#     X = [preprocess_text(text) for text in X]

#     # Aplicar TF-IDF
#     vectorizer = TfidfVectorizer(
#         min_df=5, max_df=0.9, max_features=max_features, strip_accents='unicode', lowercase=True,
#         analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1, 2),
#         use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words="english"
#     )
#     X = vectorizer.fit_transform(X)

#     # Dividir datos
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#     combined_data = {
#         "X_train": X_train, "y_train": y_train,
#         "X_val": X_val, "y_val": y_val,
#         "X_test": X_test, "y_test": y_test,
#         "vectorizer": vectorizer
#     }

#     # Guardar el archivo combinado
#     joblib.dump(combined_data, combined_file)

#     print(f"Datos y vectorizador guardados en {combined_file}")

#     return combined_data, combined_file

# def list_models_in_directory():
#     """List available models in the current directory."""
#     print("\nAvailable models:")
#     models = [f for f in os.listdir() if f.startswith('model_') and f.endswith('.joblib')]
#     if models:
#         for i, model in enumerate(models):
#             print(f"{i + 1}. {model}")
#     else:
#         print("No models found.")
#     return models

# def load_model(model_path):
#     """Load a pre-trained model."""
#     print(f"Loading model from {model_path}...")
#     return joblib.load(model_path)



# def train_model(preprocessed_data, max_features):
#     """Train a model based on user selection."""
#     print("Choose the model to train:")
#     print("1. Naive Bayes")
#     print("2. Logistic Regression")
#     print("3. Random Forest")
#     print("4. XGBoost")
#     model_choice = int(input("Select an option (1-4): "))

#     print("Do you want to perform GridSearch for hyperparameter tuning? (y/n)")
#     gridsearch = input("Your choice: ").strip().lower() == 'y'

#     if model_choice == 1:
#         from projecte_ML_train_NB import NaiveBayesModel
#         ModelClass = NaiveBayesModel
#         model_name = "NB"
#     elif model_choice == 2:
#         from projecte_ML_train_LR import LogisticRegressionModel
#         ModelClass = LogisticRegressionModel
#         model_name = "LR"
#     elif model_choice == 3:
#         from projecte_ML_train_RND_FOREST import RandomForestModel
#         ModelClass = RandomForestModel
#         model_name = "RF"
#     elif model_choice == 4:
#         from projecte_ML_train_XgBoost import XGBoostModel
#         ModelClass = XGBoostModel
#         model_name = "XgB"
#     else:
#         print("Invalid choice.")
#         return

#     print("Training the model...")

#     X_train = preprocessed_data["X_train"]
#     y_train = preprocessed_data["y_train"]
#     X_val = preprocessed_data["X_val"]
#     y_val = preprocessed_data["y_val"]
#     vectorizer = preprocessed_data["vectorizer"]

#     # Crear y entrenar el modelo con el nombre dinámico
#     model_filename = f"model_{model_name}_{max_features}.joblib"
#     model = ModelClass(X_train, y_train, vectorizer, model_path=model_filename)
#     model.train_model(optimize_model=gridsearch)
#     model.evaluar_modelo(X_val, y_val)

#     # Guardar el model complet, incloent-hi la classe
#     joblib.dump(model, model_filename)
#     print(f"Model saved as {model_filename}.")

# def test_model(preprocessed_data):
#     """Test a pre-trained model."""
#     models = list_models_in_directory()
#     model_choice = int(input("Select a model to load (by number): "))

#     if model_choice < 1 or model_choice > len(models):
#         print("Invalid choice.")
#         return

#     model_path = models[model_choice - 1]
#     model = load_model(model_path)  # Carrega tota la classe LogisticRegressionModel

#     X_test = preprocessed_data["X_test"]
#     y_test = preprocessed_data["y_test"]

#     # Verificar si el model carregat té el mètode evaluar_modelo
#     if not hasattr(model, 'evaluar_modelo'):
#         print("Loaded model does not have the 'evaluar_modelo' method.")
#         return

#     try:
#         # Cridar el mètode evaluar_modelo
#         model.evaluar_modelo(X_test, y_test)
#     except Exception as e:
#         print(f"Error during testing: {e}")

# def validate_phrase(preprocessed_data):
#     models = list_models_in_directory()
#     model_choice = int(input("Select a model to load (by number): "))

#     if model_choice < 1 or model_choice > len(models):
#         print("Invalid choice.")
#         return

#     model_path = models[model_choice - 1]
#     vectorizer_path = model_path.replace("model_", "vectorizer_")  # Asume nombres consistentes
#     model = load_model(model_path)
#     vectorizer = load_model(vectorizer_path)

#     phrase = input("Enter a phrase to classify: ")

#     try:
#         phrase_vectorized = vectorizer.transform([phrase])
#         prediction = model.predict(phrase_vectorized)
#         print(f"Prediction for the given phrase: {prediction}")
#     except Exception as e:
#         print(f"Error during phrase validation: {e}")

# dataset_path =  r"C:\Users\naman\OneDrive\Escritorio\pipeline_try\pipeline\training.1600000.processed.noemoticon.csv"

# def main():
#     max_features = int(input("Enter max_features for TF-IDF vectorizer (e.g., 10000): "))
#     lemmatization = int(input("Choose preprocessing: 1. Lemmatization 2. Stemming: ")) == 1

#     preprocessed_data, combined_file = preprocess_and_vectorize_data(dataset_path, max_features, lemmatization)

#     while True:
#         print("\nChoose an option:")
#         print("1. Train a model")
#         print("2. Test a model with a dataset")
#         print("3. Validate a single phrase")
#         print("4. Exit")
    
#         choice = int(input("Enter your choice (1-4): "))
    
#         if choice == 1:
#             train_model(preprocessed_data, max_features)
#         elif choice == 2:
#             test_model(preprocessed_data)
#         elif choice == 3:
#             validate_phrase(preprocessed_data)
#         elif choice == 4:
#             print("Exiting...")
#             break
#         else:
#             print("Invalid choice. Please try again.")


# if __name__ == "__main__":
#     main()

'''
###MAIN OFFICIAL ON VAN JA TOTS ELS ALGORITMES CORRECTAMENT###
'''


# import os
# import joblib
# import pandas as pd
# from nltk.stem import WordNetLemmatizer, PorterStemmer
# import re
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# def preprocessar_i_vectoritzar_dades(dataset_path, max_features, lemmatization=True):
#     """Preprocesa les dades, les vectoritza i desa tot en un sol fitxer."""
#     suffix = 'l' if lemmatization else 's'
#     combined_file = f"model_combined_{suffix}_{max_features}.joblib"

#     if os.path.exists(combined_file):
#         print("Carregant dades i vectoritzador des del fitxer combinat...")
#         return joblib.load(combined_file), combined_file

#     print("Preprocessant dades i aplicant TF-IDF...")
#     lemmatizer = WordNetLemmatizer()
#     stemmer = PorterStemmer()
#     column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
#     data = pd.read_csv(dataset_path, names=column_names, header=0, delimiter=",", encoding='latin1')
#     X, y = data['text'], data['target']

#     # Preprocessar els textos
#     def preprocess_text(text):
#         text = text.lower()
#         text = re.sub(r'http\S+|www\.\S+', '', text)
#         text = re.sub(r'@\w+', '', text)
#         text = re.sub(r'#', '', text)
#         text = re.sub(r'[^\w\s]', '', text)

#         if lemmatization:
#             return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
#         else:
#             return ' '.join([stemmer.stem(word) for word in text.split()])

#     X = [preprocess_text(text) for text in X]

#     # Aplicar TF-IDF
#     vectorizer = TfidfVectorizer(
#         min_df=5, max_df=0.9, max_features=max_features, strip_accents='unicode', lowercase=True,
#         analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1, 2),
#         use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words="english"
#     )
#     X = vectorizer.fit_transform(X)

#     # Dividir dades
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#     combined_data = {
#         "X_train": X_train, "y_train": y_train,
#         "X_val": X_val, "y_val": y_val,
#         "X_test": X_test, "y_test": y_test,
#         "vectorizer": vectorizer
#     }

#     # Desa el fitxer combinat
#     joblib.dump(combined_data, combined_file)

#     print(f"Dades i vectoritzador desats a {combined_file}")

#     return combined_data, combined_file

# def llistar_models_al_directori():
#     """Llista els models disponibles al directori actual."""
#     print("\nModels disponibles:")
#     models = [f for f in os.listdir() if f.startswith('model_') and f.endswith('.joblib')]
#     if models:
#         for i, model in enumerate(models):
#             print(f"{i + 1}. {model}")
#     else:
#         print("No s'han trobat models.")
#     return models

# def carregar_model(model_path):
#     """Carrega un model entrenat."""
#     print(f"Carregant model des de {model_path}...")
#     return joblib.load(model_path)

# def entrenar_model(dades_preprocessades, max_features):
#     """Entrena un model segons l'opció seleccionada per l'usuari."""
#     print("Escull el model a entrenar:")
#     print("1. Naive Bayes")
#     print("2. Regressió Logística")
#     print("3. Random Forest")
#     print("4. XGBoost")
#     model_choice = int(input("Selecciona una opció (1-4): "))

#     print("Vols fer servir GridSearch per ajustar els hiperparàmetres? (s/n)")
#     gridsearch = input("La teva elecció: ").strip().lower() == 's'

#     if model_choice == 1:
#         from projecte_ML_train_NB import NaiveBayesModel
#         ModelClass = NaiveBayesModel
#         model_name = "NB"
#     elif model_choice == 2:
#         from projecte_ML_train_LR import LogisticRegressionModel
#         ModelClass = LogisticRegressionModel
#         model_name = "LR"
#     elif model_choice == 3:
#         from projecte_ML_train_RND_FOREST import RandomForestModel
#         ModelClass = RandomForestModel
#         model_name = "RF"
#     elif model_choice == 4:
#         from projecte_ML_train_XgBoost import XGBoostModel
#         ModelClass = XGBoostModel
#         model_name = "XgB"
#     else:
#         print("Elecció no vàlida.")
#         return

#     print("Entrenant el model...")

#     X_train = dades_preprocessades["X_train"]
#     y_train = dades_preprocessades["y_train"]
#     X_val = dades_preprocessades["X_val"]
#     y_val = dades_preprocessades["y_val"]
#     vectorizer = dades_preprocessades["vectorizer"]

#     model_filename = f"model_{model_name}_{max_features}.joblib"
#     model = ModelClass(X_train, y_train, vectorizer, model_path=model_filename)
#     model.train_model(optimize_model=gridsearch)
#     model.evaluar_modelo1(X_val, y_val)

#     joblib.dump(model, model_filename)
#     print(f"Model desat com {model_filename}.")

# def provar_model(dades_preprocessades):
#     """Prova un model entrenat."""
#     models = llistar_models_al_directori()
#     model_choice = int(input("Selecciona un model a carregar (per número): "))

#     if model_choice < 1 or model_choice > len(models):
#         print("Elecció no vàlida.")
#         return

#     model_path = models[model_choice - 1]
#     model = carregar_model(model_path)

#     X_test = dades_preprocessades["X_test"]
#     y_test = dades_preprocessades["y_test"]

#     if not hasattr(model, 'evaluar_model'):
#         print("El model carregat no té el mètode 'evaluar_model'.")
#         return

#     try:
#         model.evaluar_modelo(X_test, y_test)
#     except Exception as e:
#         print(f"Error durant la prova: {e}")

# def validar_frase(dades_preprocessades):
#     models = llistar_models_al_directori()
#     model_choice = int(input("Selecciona un model a carregar (per número): "))

#     if model_choice < 1 or model_choice > len(models):
#         print("Elecció no vàlida.")
#         return

#     model_path = models[model_choice - 1]
#     model = carregar_model(model_path)

#     frase = input("Introdueix una frase per classificar: ")

#     try:
#         vectorizer = dades_preprocessades["vectorizer"]
#         frase_vectoritzada = vectorizer.transform([frase])
#         prediccio = model.predict(frase_vectoritzada)[0]  # Obtenim la predicció com a valor únic

#         # Interpretació del resultat
#         if prediccio == 0:
#             resultat = "negativa"
#         elif prediccio == 4:
#             resultat = "positiva"
#         else:
#             resultat = f"desconeguda (predicció: {prediccio})"

#         print(f"La frase introduïda és {resultat}.")
#     except Exception as e:
#         print(f"Error durant la validació de la frase: {e}")

# def main():
#     dataset_path = r"C:\Users\naman\OneDrive\Escritorio\pipeline_try\pipeline\training.1600000.processed.noemoticon.csv"
#     max_features = int(input("Introdueix max_features per al vectoritzador TF-IDF (p. ex., 10000): "))
#     lemmatization = int(input("Escull el preprocessament: 1. Lemmatització 2. Stemming: ")) == 1

#     dades_preprocessades, combined_file = preprocessar_i_vectoritzar_dades(dataset_path, max_features, lemmatization)

#     while True:
#         print("\nEscull una opció:")
#         print("1. Entrenar un model")
#         print("2. Provar un model amb un conjunt de dades")
#         print("3. Validar una frase")
#         print("4. Sortir")

#         choice = int(input("Introdueix la teva elecció (1-4): "))

#         if choice == 1:
#             entrenar_model(dades_preprocessades, max_features)
#         elif choice == 2:
#             provar_model(dades_preprocessades)
#         elif choice == 3:
#             validar_frase(dades_preprocessades)
#         elif choice == 4:
#             print("Sortint...")
#             break
#         else:
#             print("Elecció no vàlida. Torna-ho a intentar.")

# if __name__ == "__main__":
#     main()



###PROVA DE MAIN NANA##

import os
import joblib
import pandas as pd
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from projecte_ML_train_LR import LogisticRegressionModel
from projecte_ML_train_NB import NaiveBayesModel
from projecte_ML_train_RND_FOREST import RandomForestModel
from projecte_ML_train_XgBoost import XGBoostModel

def preprocessar_i_vectoritzar_dades(dataset_path, max_features, lemmatization=True):
    """Preprocesa les dades, les vectoritza i desa tot en un sol fitxer."""
    suffix = 'l' if lemmatization else 's'
    combined_file = f"model_combined_{suffix}_{max_features}.joblib"

    if os.path.exists(combined_file):
        print("Carregant dades i vectoritzador des del fitxer combinat...")
        return joblib.load(combined_file), combined_file

    print("Preprocessant dades i aplicant TF-IDF...")
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
    data = pd.read_csv(dataset_path, names=column_names, header=0, delimiter=",", encoding='latin1')
    X, y = data['text'], data['target']

    # Preprocessar els textos
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'[^\w\s]', '', text)

        if lemmatization:
            return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        else:
            return ' '.join([stemmer.stem(word) for word in text.split()])

    processed_data = [(preprocess_text(text), label) for text, label in zip(X, y) if preprocess_text(text).strip()]
    X, y = zip(*processed_data)  # Desempaquetar dades processades

    # Aplicar TF-IDF
    vectorizer = TfidfVectorizer(
        min_df=5, max_df=0.9, max_features=max_features, strip_accents='unicode', lowercase=True,
        analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1, 2),
        use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words="english"
    )
    X = vectorizer.fit_transform(X)

    # Dividir dades
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)  # 70% per entrenament
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 15% validació, 15% test

    combined_data = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "vectorizer": vectorizer
    }

    # Desa el fitxer combinat
    joblib.dump(combined_data, combined_file)

    print(f"Dades i vectoritzador desats a {combined_file}")

    return combined_data, combined_file

def validar_frase(model, vectorizer):
    """Valida una frase amb un model carregat."""
    frase = input("Introdueix una frase per classificar: ")
    frase_vectoritzada = vectorizer.transform([frase])

    try:
        if hasattr(model, 'predict'):
            prediccio = model.predict(frase_vectoritzada)[0]
            resultat = "positiva" if prediccio == 1 else "negativa"
            print(f"La frase introduïda és {resultat}.")
        else:
            print("El model carregat no suporta la predicció de frases.")
    except Exception as e:
        print(f"Error durant la validació de la frase: {e}")

def main():
    dataset_path = r"C:\Users\naman\OneDrive\Escritorio\pipeline_try\pipeline\training.1600000.processed.noemoticon.csv"
    max_features = int(input("Introdueix max_features per al vectoritzador TF-IDF (p. ex., 10000): "))
    lemmatization = int(input("Escull el preprocessament: 1. Lemmatització 2. Stemming: ")) == 1

    dades_preprocessades, combined_file = preprocessar_i_vectoritzar_dades(dataset_path, max_features, lemmatization)

    while True:
        print("\nEscull una opció:")
        print("1. Entrenar un model")
        print("2. Provar un model amb un conjunt de dades")
        print("3. Validar una frase")
        print("4. Sortir")

        choice = int(input("Introdueix la teva elecció (1-4): "))

        if choice == 1:
            print("Escull el model a entrenar:")
            print("1. Naive Bayes")
            print("2. Regressió Logística")
            print("3. Random Forest")
            print("4. XGBoost")
            model_choice = int(input("Selecciona una opció (1-4): "))

            print("Vols fer servir GridSearch per ajustar els hiperparàmetres? (s/n)")
            gridsearch = input("La teva elecció: ").strip().lower() == 's'

            X_train = dades_preprocessades["X_train"]
            y_train = dades_preprocessades["y_train"]
            X_val = dades_preprocessades["X_val"]
            y_val = dades_preprocessades["y_val"]
            vectorizer = dades_preprocessades["vectorizer"]

            if model_choice == 1:
                model = NaiveBayesModel(X_train, y_train, vectorizer)
            elif model_choice == 2:
                model = LogisticRegressionModel(X_train, y_train, vectorizer)
            elif model_choice == 3:
                model = RandomForestModel(X_train, y_train, vectorizer)
            elif model_choice == 4:
                model = XGBoostModel(X_train, y_train, vectorizer)
            else:
                print("Elecció no vàlida.")
                continue

            model.train_model(X_val, y_val, optimize_model=gridsearch)

        elif choice == 2:
            models = [f for f in os.listdir() if f.startswith('model_') and f.endswith('.joblib')]
            print("\nModels disponibles:")
            for i, model_file in enumerate(models):
                print(f"{i + 1}. {model_file}")

            model_choice = int(input("Selecciona un model a carregar (per número): "))
            if model_choice < 1 or model_choice > len(models):
                print("Elecció no vàlida.")
                continue

            model_path = models[model_choice - 1]
            print(f"Carregant model des de {model_path}...")
            model = joblib.load(model_path)

            if hasattr(model, 'evaluar_modelo'):
                X_test = dades_preprocessades["X_test"]
                y_test = dades_preprocessades["y_test"]
                model.evaluar_modelo(X_test, y_test)
            else:
                print("El model carregat no suporta la funció d'avaluació.")

        elif choice == 3:
            models = [f for f in os.listdir() if f.startswith('model_') and f.endswith('.joblib')]
            print("\nModels disponibles:")
            for i, model_file in enumerate(models):
                print(f"{i + 1}. {model_file}")

            model_choice = int(input("Selecciona un model a carregar (per número): "))
            if model_choice < 1 or model_choice > len(models):
                print("Elecció no vàlida.")
                continue

            model_path = models[model_choice - 1]
            print(f"Carregant model des de {model_path}...")
            model = joblib.load(model_path)

            validar_frase(model, dades_preprocessades["vectorizer"])

        elif choice == 4:
            print("Sortint...")
            break
        else:
            print("Elecció no vàlida. Torna-ho a intentar.")

if __name__ == "__main__":
    main()
