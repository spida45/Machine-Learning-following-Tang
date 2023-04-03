import numpy as np
from utils.features import prepare_for_training


class LinearRegression:

    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        1. data pre_processing
        2. get number of features
        3. initialize parameter matrix
        """

        # Normalize features and add ones column.
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        # Initialize model parameters.
        num_features = self.data.shape[1]   # number of features is just number of columns
        self.theta = np.zeros((num_features, 1))    # theta is corresponding to feature

    def train(self, alpha, num_interactions=500):
        """
        训练模块：执行梯度下降
        """
        cost_history = self.gradient_decent(alpha, num_interactions)

        return self.theta, cost_history

    def gradient_decent(self, alpha, num_interactions):
        """
        实际迭代模块：迭代num_interactions
        """
        cost_history = []
        for _ in range(num_interactions):
            self.gradient_step(alpha)

            cost_history.append(self.cost_function(self.data, self.labels))

        return cost_history

    def gradient_step(self, alpha):
        """
        梯度下降参数更新方法，注意是矩阵运算
        """
        num_examples = self.data.shape[0]
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        delta = prediction - self.labels
        theta = self.theta
        theta = theta - alpha * (1 / num_examples) * (np.dot(delta.T, self.data)).T
        self.theta = theta

    @staticmethod
    def hypothesis(data, theta):
        prediction = np.dot(data, theta)
        return prediction

    def cost_function(self, data, labels):
        """
        损失函数计算方法
        """
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(self.data, self.theta) - labels
        cost = (1 / 2) * np.dot(delta.T, delta)
        # 自己定义损失函数时不知道要返回矩阵中的哪个值，可以先打印cost的shape值，然后再打印一下cost
        # print(cost.shape)
        return cost[0][0]

    def get_cost(self, data, labels):
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]

        return self.cost_function(data_processed, labels)

    def predict(self, data):
        """
        用训练好的参数模型，预测得到回归值结果
        """
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]
        predictions = LinearRegression.hypothesis(data_processed, self.theta)

