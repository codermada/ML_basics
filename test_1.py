import pandas as pd
import matplotlib.pyplot as plt


class Model:
    def __init__(self):
        df = pd.read_csv("data_1.csv")
        self.x = df['x']
        self.y = df['y']
        self.w = 0
    def plot(self):
        plt.scatter(self.y, self.x)
        plt.show()
    def mse(self, w):
        y_ = self.x * w
        d = y_ - self.y
        n = len(d)
        return sum(d*d) / n
    def dcost(self, eps):
        return (self.mse(self.w + eps) - self.mse(self.w)) / eps
    def train(self, rate, n):
        for i in range(n):
            dcost = self.dcost(0.001)
            self.w = self.w - rate * dcost
            # print(dcost, self.w)
    def predict(self, x_):
        return x_ * self.w


model = Model()
# model.plot()
# print(model.dcost(0.001))
model.train(0.001, 20000)
print(model.predict(9))