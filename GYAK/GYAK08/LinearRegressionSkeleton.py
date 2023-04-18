import numpy as np


class LinearRegression:
    def __init__(self, epochs: int = 1000, lr: float = 1e-3):
        self.epochs = epochs
        self.lr = lr

    def fit(self, X: np.array, y: np.array):
    # inicializáljuk a súlyokat, bias-t és a költségfüggvényt
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        self.cost = []

        # gradient descent algoritmus futtatása az epochs számának megfelelően
        for epoch in range(self.epochs):
            # kiszámítjuk a predikciókat a jelenlegi súlyok és bias alapján
            y_pred = np.dot(X, self.weights) + self.bias

            # számítjuk a költségfüggvény értékét és eltároljuk
            cost = np.mean((y_pred - y) ** 2)
            self.cost.append(cost)

            # számítjuk a gradienteket a súlyok és bias szerint
            dw = np.dot(X.T, (y_pred - y)) / len(y)
            db = np.sum(y_pred - y) / len(y)

            # frissítjük a súlyokat és bias-t a gradient descent algoritmus alapján
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        

    def predict(self, X):
        # kiszámítjuk a predikciókat a tanult súlyok és bias alapján
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

