

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
