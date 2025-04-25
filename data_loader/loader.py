# cardio_library/data_loader/loader.py

import pandas as pd

def load_cardio_data(path):
    """
    Загружает датасет cardio_train.csv
    """
    df = pd.read_csv(path, sep=';')
    print(f"Датасет загружен: {df.shape[0]} строк, {df.shape[1]} столбцов.")
    return df
