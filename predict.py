import pandas as pd
from utils.save_load import load_model, load_transformer

def main():
    input_path = "data/new_data.csv"
    df_new = pd.read_csv(input_path, sep=';')

    transformer = load_transformer()
    model = load_model()

    df_cleaned = transformer.clean_data(df_new)
    df_scaled = transformer.transform(df_cleaned)

    predictions = model.predict(df_scaled)
    print("Предсказания (0 — нет болезни, 1 — есть болезнь):")
    print(predictions)

if __name__ == "__main__":
    main()
