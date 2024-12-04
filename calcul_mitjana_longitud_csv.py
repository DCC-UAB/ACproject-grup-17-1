import pandas as pd

def calcula_mitjana_longitud_textos(csv_file):
    try:
        # Llegim el fitxer CSV
        column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
        data = pd.read_csv(csv_file, names=column_names, header=0, delimiter=",", encoding='latin1')
        
        # Comprovem que existeixi la columna `text`
        if 'text' not in data.columns:
            print("Error: No s'ha trobat la columna 'text' al fitxer.")
            return None

        # Calculem les longituds dels texts
        data['text_length'] = data['text'].str.len()
        
        # Calculem la mitjana
        mitjana = data['text_length'].mean()
        print(f"La mitjana de les longituds dels texts és: {mitjana:.2f} caràcters.")
        return mitjana
    except FileNotFoundError:
        print(f"Error: Fitxer {csv_file} no trobat.")
    except Exception as e:
        print(f"Error: {e}")

# Exemple d'ús
csv_file_path = r"C:\Users\naman\OneDrive\Escritorio\3r\Projecte_ML_twit_Nana\training.1600000.processed.noemoticon.csv"
calcula_mitjana_longitud_textos(csv_file_path)
