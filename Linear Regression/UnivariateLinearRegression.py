import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression
# from spacy.tests.regression.test_issue910 import train_data

data = pd.read_csv('../data/world-happiness-report-2017.csv')

# 得到训练和测试数据
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[input_param_name].values
y_test = test_data[output_param_name].values

plt.scatter(x_train, y_train, label='Train data')
plt.scatter(x_test, y_test, label='Test data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happiness Score')
plt.legend()
plt.show()

num_interactions = 5000
learning_rate = 0.001

linear_regression = LinearRegression(x_train, y_train)
(theta, cost_history) = linear_regression.train(learning_rate, num_interactions)

print('开始时的损失：', cost_history[0])
print('训练后的损失：', cost_history[-1])

plt.plot(range(num_interactions), cost_history)
plt.xlabel('Iter')
plt.ylabel('Cost')
plt.title('Gradient Descent')
plt.show()
