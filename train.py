from data_loader.loader import load_cardio_data
from preprocessing.transformer import DataTransformer
from models.logistic_model import LogisticModel
from metrics.evaluation import evaluate_model
from utils.save_load import save_model, save_transformer
from visualization.plots import plot_roc_curve, plot_confusion, plot_learning_curve

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# 1. Загрузка данных
df = load_cardio_data("data/cardio_train.csv")

# 2. Предобработка
transformer = DataTransformer()
df = transformer.clean_data(df)
df_scaled = transformer.fit_transform(df)

# 3. Разделение на X и y
X = df_scaled.drop(columns=['cardio'])
y = df_scaled['cardio']

# 4. Деление на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Обучение модели
model = LogisticModel()
model.fit(X_train, y_train)

# 6. Предсказания
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# 7. Оценка качества
evaluate_model(y_test, y_pred, y_proba)

# 8. Визуализация
plot_roc_curve(y_test, y_proba)
plot_confusion(y_test, y_pred)
plot_learning_curve(model.model, X, y)

# 9. Сохранение модели и трансформера
from utils.save_load import save_model, save_transformer

# Сохраняем обученную модель и трансформер
save_model(model)
save_transformer(transformer)
