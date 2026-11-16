import pandas as pd
import matplotlib.pyplot as plt


class Model:
    def __init__(self):
        df = pd.read_csv("data_2.csv")
        self.x = df['x']
        self.y = df['y']
        self.w = 0
        self.b = 1
    def plot(self):
        plt.scatter(self.y, self.x)
        plt.show()
    def mse(self, w, b):
        y_ = self.x * w + b
        d = y_ - self.y
        n = len(d)
        return sum(d*d) / n
    def dw(self, eps, b):
        return (self.mse(self.w + eps, b) - self.mse(self.w, b)) / eps
    def db(self, eps, b):
        return (self.mse(self.w , b + eps) - self.mse(self.w, b)) / eps
    def train(self, rate, n):
        for i in range(n):
            dw = self.dw(0.01, self.b)
            db = self.db(0.01, self.b)
            self.w = self.w - rate * dw
            self.b = self.b - rate * db
            # print(dcost, self.w)
    def predict(self, x_):
        return x_ * self.w
    def __str__(self):
        return f"w = {self.w}, b = {self.b}"


model = Model()
# model.plot()
# print(model.dcost(0.001))
model.train(0.001, 15000)
print(model.predict(2))
print(model)