import numpy as np
import matplotlib.pyplot as plt
import time


# 读取数据
def read_data(path):
    data = np.loadtxt(path)
    label = data[:, -1]
    data = data[:, :-1]
    return data, label


# 数据标准化
def data_normalization(data):
    data_normed = data
    for i in range(data.shape[1]):
        cur_col = data[:, i]
        mean = np.mean(cur_col)
        std = np.std(cur_col)
        for j in range(cur_col.shape[0]):
            data_normed[j, i] = (data[j, i] - mean) / std
    return data_normed


# 留一法划分
def loo_split(data, label, idx):
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    for i in range(0, idx):
        train_data.append(data[i])
        train_label.append(label[i])
    test_data.append(data[idx])
    test_label.append(label[idx])
    for i in range(idx+1, data.shape[0]):
        train_data.append(data[i])
        train_label.append(label[i])
    return np.array(train_data), np.array(train_label), \
           np.array(test_data), np.array(test_label)


def gradient(weight, x, y, random=False):
    if random is False:
        error = np.dot(x, weight) - y
        return abs(error.mean()), (1/x.shape[0]) * np.dot(np.transpose(x), error)  # 注意各个矩阵维度的匹配
    else:
        rand = np.random.randint(0, x.shape[0] - 1)  # 用numpy中的方法生成随机数
        x_rand = x[rand, :]
        y_rand = y[rand]
        error = np.dot(x_rand, weight) - y_rand
        return abs(error.mean()), (1/x.shape[0]) * np.dot(x_rand, error)


def gradient_L1(weight, x, y, alpha, random=False):
    if random is False:
        error = np.dot(x, weight) - y
        return abs(error.mean()), (1/x.shape[0]) * (np.dot(np.transpose(x), error)
                                                    + alpha * np.sign(weight))  # 正则项求导后为sign(x)
    else:
        rand = np.random.randint(0, x.shape[0] - 1)
        x_rand = x[rand, :]
        y_rand = y[rand]
        error = np.dot(x_rand, weight) - y_rand
        return abs(error.mean()), (1/x.shape[0]) * (np.dot(x_rand, error) + alpha * np.sign(weight))


def gradient_L2(weight, x, y, alpha, random=False):
    if random is False:
        error = np.dot(x, weight) - y
        # 正则项求导后为weight值的一次形式
        return abs(error.mean()), (1/x.shape[0]) * (np.dot(np.transpose(x), error) + alpha * weight)
    else:
        rand = np.random.randint(0, x.shape[0] - 1)
        x_rand = x[rand, :]
        y_rand = y[rand]
        error = np.dot(x_rand, weight) - y_rand
        return abs(error.mean()), (1/x.shape[0]) * (np.dot(x_rand, error) + alpha * weight)


def gradient_descent(data, label, lr, epochs, reg=None, alpha=None, random=False):
    weight = np.zeros((14))  # 在13个特征列基础上加一个常数项
    x = np.hstack((np.ones((data.shape[0], 1)), data))  # 给数据加上全1的一列
    y = label
    errors = []
    if reg is None:
        for epoch in range(epochs):
            error, grad = gradient(weight, x, y, random)
            weight = weight - lr * grad
            errors.append(error)
    elif reg is 'L1':
        for epoch in range(epochs):
            error, grad = gradient_L1(weight, x, y, alpha, random)
            weight = weight - lr * grad
            errors.append(error)
    else:
        for epoch in range(epochs):
            error, grad = gradient_L2(weight, x, y, alpha, random)
            weight = weight - lr * grad
            errors.append(error)
    return errors, weight


def linear_regression(data, label, lr, epochs, reg=None, alpha=None, random=False):
    data = data_normalization(data)
    preds = []
    for i in range(data.shape[0]):  # 每一条数据都要用留一法进行预测
        train_data, train_label, test_data, test_label = loo_split(data, label, i)
        errors, weight = gradient_descent(train_data, train_label, lr, epochs, reg, alpha, random)
        test_x = np.hstack((np.array([1]).reshape((1, 1)), test_data))
        pred_y = np.dot(test_x, weight)
        preds.append(pred_y)
        if i % 100 == 0:
            print(i, 'loo')
    preds = np.array(preds).reshape(label.shape)
    rmse = np.sqrt(((preds - label) ** 2).mean())  # 计算所有预测值的RMSE
    return preds, rmse, errors


data_path = 'housing.data'
data, label = read_data(data_path)
lr = 0.1
epochs = 1000
alpha = 0.1
t1 = time.time()
pred, rmse, errors = linear_regression(data, label, lr, epochs)
t2 = time.time()
pred_L1, rmse_L1, errors_L1 = linear_regression(data, label, lr, epochs, 'L1', alpha)
t3 = time.time()
pred_L2, rmse_L2, errors_L2 = linear_regression(data, label, lr, epochs, 'L2', alpha)
t4 = time.time()
pred_random, rmse_random, errors_random = linear_regression(data, label, lr, epochs, random=True)
t5 = time.time()
pred_L1_random, rmse_L1_random, errors_L1_random = linear_regression(data, label, lr, epochs,
                                                                     'L1', alpha, random=True)
t6 = time.time()
pred_L2_random, rmse_L2_random, errors_L2_random = linear_regression(data, label, lr, epochs,
                                                                     'L2', alpha, random=True)
t7 = time.time()

print('rmse gradient descent:', rmse)
print('time cost gradient descent:', t2 - t1, 's')
print('rmse L1 regularization gradient descent:', rmse_L1)
print('time cost L1 regularization gradient descent:', t3 - t2, 's')
print('rmse L2 regularization gradient descent:', rmse_L2)
print('time cost L2 regularization gradient descent:', t4 - t3, 's')
print('rmse random gradient descent:', rmse_random)
print('time cost random gradient descent:', t5 - t4, 's')
print('rmse L1 regularization random gradient descent:', rmse_L1_random)
print('time cost L1 regularization random gradient descent:', t6 - t5, 's')
print('rmse L2 regularization random gradient descent:', rmse_L2_random)
print('time cost L2 regularization random gradient descent:', t7 - t6, 's')

k = [x for x in range(1, 101)]

plt.title('GD vs. SGD')
plt.plot(k, errors[0:100], color='green', label='gradient descent loss')
plt.plot(k, errors_random[0:100], color='red', label='random gradient descent loss')
plt.legend()

plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()