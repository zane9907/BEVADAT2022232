import numpy as np


class LinearRegression:
    def __init__(self, epochs: int = 1000, lr: float = 1e-3):
        self.epochs = epochs
        self.lr = lr

    def fit(self, X: np.array, y: np.array):
        n_samples = X.shape[0]
        n_features = X.shape[1]

        self.weights = np.zeros(n_features + 1)

        X = np.insert(X, 0, 1, axis=1)

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights)
            error = y - y_pred
            gradient = -2 * np.dot(X.T, error) / n_samples
            self.weights -= self.lr * gradient
        
   

        

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = np.dot(X, self.weights)
        return y_pred
        
