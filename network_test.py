""" 自己写一个网络模型,学习前向传播、反向传播和梯度下降法,从零搭建一个神经网络我们需要四个步骤：
1、定义网络的结构，包含输入，输出，隐藏层
2、初始化模型的参数
3、正向传播（计算损失值）
4、反向传播（对权值进行更新）
"""

import numpy as np
import matplotlib.pyplot as plt

stru=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(stru)


def sigmoid(x):
    a = 1 / (1 + np.exp(-x))
    return a


def initilize(y):
    w = np.zeros((y, 1))
    b = 0.0
    return w, b


def forward_propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)  # 激活函数
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # 计算损失

    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m
    # 最好要断言一下数据类型
    # assert(dw.shape == w.shape)
    # assert(db.dtype == float)

    cost = np.squeeze(cost)
    # assert(cost.shape == ())

    grads = {'dw': dw, 'db': db}

    return grads, cost


def backward_propagation(w, b, X, Y, num, learning_rate, print_cost=False):
    """
    param:
    num:迭代的次数
    learning_rate：学习率
    print_cost：损失

    return:
    params, grads, cost
    """
    cost = []
    for i in range(num):
        grad, cost = forward_propagate(w, b, X, Y)

        dw = grad['dw']
        db = grad['db']

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            cost.append(cost)
        if print_cost and i % 100 == 0:
            print("cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b
              }

    grads = {"dw": dw,
             "db": db
             }

    return params, grads, cost


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[:, i] > 0.5:
            Y_prediction[:, i] = 1
        else:
            Y_prediction[:, i] = 0

    assert (Y_prediction.shape == (1, m))
    return Y_prediction


# 损失函数
def loss(y, y_):
    """
    :param y:  样本值
    :param y_: 预测值
    :param m:  样本数量
    :return: 损失值
    """
    return np.sum(np.square(y - y_))


# 训练数据，一个sin函数
sample_x = np.linspace(0, 10, 100)
sample_y = np.sin(sample_x) + 2

# 搭建模型，三层输入，隐藏层，输出层
# 输入层
x = sample_x

# 添加隐藏层,h1,10个神经元,w1的数量=100 * 10
w1 = np.random.rand(100, 10)
b1 = np.zeros((10, ))
net1 = np.dot(w1.T, x) + b1
out1 = sigmoid(net1)

# 输出层,2个神经元
w2 = np.random.rand(10, 2)
b2 = np.zeros((2,))
net2 = np.dot(w2.T, out1) + b2
out2 = sigmoid(net2)

loss = loss(sample_y - y_)