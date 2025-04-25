# cardio_library/visualization/plots.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
import numpy as np

def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='ROC-кривая')
    plt.plot([0, 1], [0, 1], 'k--', label='Случайный выбор')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Матрица ошибок")
    plt.grid(False)
    plt.show()

def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5,
                                                            scoring='accuracy',
                                                            train_sizes=np.linspace(0.1, 1.0, 5))
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, label="Train score")
    plt.plot(train_sizes, test_mean, label="Validation score")
    plt.xlabel("Размер обучающей выборки")
    plt.ylabel("Точность")
    plt.title("Кривая обучения")
    plt.legend()
    plt.grid(True)
    plt.show()
