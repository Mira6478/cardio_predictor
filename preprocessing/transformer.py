import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataTransformer:
    def __init__(self):
        self.scaler = StandardScaler()

    def clean_data(self, df):
        # Перевод возраста в годы
        df['age'] = (df['age'] / 365).round().astype(int)

        # Удаление аномалий по давлению
        df = df[(df['ap_hi'] >= df['ap_lo']) & (df['ap_hi'] < 250) & (df['ap_lo'] > 40)]

        return df

    def fit_transform(self, df):
        features_to_scale = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']

        df_scaled = df.copy()
        df_scaled[features_to_scale] = self.scaler.fit_transform(df_scaled[features_to_scale])

        return df_scaled

    def transform(self, df):
        features_to_scale = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']

        df_scaled = df.copy()
        df_scaled[features_to_scale] = self.scaler.transform(df_scaled[features_to_scale])

        return df_scaled
