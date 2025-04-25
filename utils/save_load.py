import joblib
import os

def save_model(model, path="models/logistic_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Модель сохранена в {path}")

def load_model(path="models/logistic_model.pkl"):
    if os.path.exists(path):
        print(f"Модель загружена из {path}")
        return joblib.load(path)
    else:
        raise FileNotFoundError(f"Файл {path} не найден")

def save_transformer(transformer, path="preprocessing/transformer.pkl"):
    joblib.dump(transformer, path)
    print(f"Трансформер сохранён в {path}")

def load_transformer(path="preprocessing/transformer.pkl"):
    if os.path.exists(path):
        print(f"Трансформер загружен из {path}")
        return joblib.load(path)
    else:
        raise FileNotFoundError(f"Файл {path} не найден")
