import pandas as pd
import matplotlib.pyplot as plt
from numpy import exp


def sigmoid(x):
    return 1 / (1 + exp(-x))

class Model:
    def __init__(self):
        df = pd.read_csv("data_3.csv")
        self.x1 = df['x1']
        self.x2 = df['x2']
        self.y = df['y']
        self.w1 = 0
        self.w2 = 0
        self.b = 1
    def plot(self):
        plt.scatter(self.y, self.x)
        plt.show()
    def mse(self, w1, w2, b):
        y_ = sigmoid(self.x1 * w1 + self.x2 * w2 + b)
        d = y_ - self.y
        n = len(d)
        return sum(d*d) / n
    def train(self, rate, n):
        for i in range(n):
            eps = 0.001
            dw1 = (self.mse(self.w1 + eps, self.w2, self.b) - self.mse(self.w1, self.w2, self.b)) / eps
            dw2 = (self.mse(self.w1 , self.w2 + eps, self.b) - self.mse(self.w1, self.w2, self.b)) / eps
            db = (self.mse(self.w1, self.w2 , self.b + eps) - self.mse(self.w1, self.w2, self.b)) / eps
            self.w1 = self.w1 - rate * dw1
            self.w2 = self.w2 - rate * dw2
            self.b = self.b - rate * db
    def predict(self, x1_, x2_):
        return sigmoid(x1_ * self.w1 + x2_ * self.w2 + self.b)
    def __str__(self):
        return f"w1 = {self.w1}, w2 = {self.w2}, b = {self.b}"


model = Model()
# model.plot()
# print(model.dcost(0.001))
model.train(0.05, 20000)
print(model.predict(0, 0))
print(model.predict(0, 1))
print(model.predict(1, 0))
print(model.predict(1, 1))
print(model)