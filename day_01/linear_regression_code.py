import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self,l_rate=0.01,epoch=1000):
        self.l_rate=l_rate
        self.epoch=epoch
        self.costs=[]

    def cost_func(self,m,c,x,y):
        y_pred=m*x+c
        error=y_pred-y
        return np.sum(error*error)/(2*x.size)
    
    def gradients(self,m,c,x,y):
        y_pred=m*x+c
        error=y_pred-y
        n=y.size
        dm=(1/n)*np.sum(error*x)
        dc=(1/n)*np.sum(error)
        return dm,dc
    
    def train(self,x,y):
        self.m=0
        self.c=0
        i=0
        for i in range(self.epoch):
            dm,dc=self.gradients(self.m,self.c,x,y)
            self.m-=self.l_rate*dm
            self.c-=self.l_rate*dc
            cost = self.cost_func(self.m, self.c, x, y)
            self.costs.append(cost)
            if i % 500 == 0:
               print(f"Epoch {i}: m = {self.m:.4f}, c = {self.c:.4f}, Cost = {cost:.4f}")

    def predict(self, x):
        return self.m * x + self.c

# testing it on some data
data = pd.read_csv("study_score.csv")
x=data["Hours"].values
y=data["Score"].values

#plotting data
plt.scatter(x,y)
plt.title("Data plot")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.grid()
plt.savefig("data.png")

#training model
l1=LinearRegression(l_rate=0.002, epoch=5000)
l1.train(x,y) 

#plotting predicted value

x_plot=np.linspace(0, max(x), 100)
y_plot=l1.predict(x_plot)
plt.plot(x_plot, y_plot, color='red', label='Prediction of linear regression')

plt.xlabel("Hours")
plt.ylabel("Score")
plt.title("Data + Fitted Line")
plt.legend()
plt.savefig("data_fitting.png")