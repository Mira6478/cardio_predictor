# cardio_library/models/logistic_model.py

from sklearn.linear_model import LogisticRegression

class LogisticModel:
    def __init__(self, learning_rate=0.01, reg_lambda=1.0, max_iter=1000):
        self.model = LogisticRegression(
            C=1/reg_lambda,  # регуляризация (обратная сила)
            solver='lbfgs',
            max_iter=max_iter
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
