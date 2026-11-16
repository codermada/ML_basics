import pandas as pd
import matplotlib.pyplot as plt


class Model:
    def __init__(self):
        df = pd.read_csv("data.csv")
        self.x = df['x']
        self.y = df['y']
        self.w = 0
    def plot(self):
        plt.scatter(self.y, self.x)
        plt.show()
    def mse(self):
        y_ = self.x * self.w
        d = y_ - self.y
        n = len(d)
        return sum(d*d) / n

model = Model()
# model.plot()
print(model.mse())