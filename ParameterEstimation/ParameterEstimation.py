import pandas as pd
import numpy as np
from sklearn.neighbors import kde
import matplotlib.pyplot as plt


def read_data(path):
    data = np.array(pd.read_csv(data_path, header=None))
    label = data[:, -1]
    data = data[:, :-1]
    return data, label


def train_valid_split(data, label, fold, idx):
    sample1 = data[:50]
    label1 = label[:50]
    sample2 = data[50:100]
    label2 = label[50:100]
    sample3 = data[100:]
    label3 = label[100:]
    train_idx = []
    valid_sample = []
    valid_label = []
    for i in range(sample1.shape[0]):
        if i%fold == idx:
            valid_sample.append(sample1[i, :])
            valid_sample.append(sample2[i, :])
            valid_sample.append(sample3[i, :])
            valid_label.append(label1[i])
            valid_label.append(label2[i])
            valid_label.append(label3[i])
        else:
            train_idx.append(i)
    train_sample1 = sample1[train_idx, :]
    train_sample2 = sample2[train_idx, :]
    train_sample3 = sample3[train_idx, :]
    train_label1 = label1[train_idx]
    train_label2 = label2[train_idx]
    train_label3 = label3[train_idx]
    valid_sample = np.array(valid_sample)
    valid_label = np.array(valid_label)
    return train_sample1, train_sample2, train_sample3, \
           train_label1, train_label2, train_label3, \
           valid_sample, valid_label


def cal_para(sample):
    dim = sample.shape[1]
    mean = np.mean(sample, 0)
    sigma = np.cov(sample.T)
    det_sigma = np.linalg.det(sigma)
    return dim, mean, sigma, det_sigma


def cal_prob(vec, sample):
    dim, mean, sigma, det_sigma = cal_para(sample)
    # 正态分布概率密度函数
    prob = (1 / (((2*np.pi) ** (dim/2)) * np.sqrt(det_sigma))) * \
           np.exp((-(1/2) * (vec - mean)).dot(np.mat(sigma).I).dot(vec - mean))
    return prob


def cal_prob_smooth(vec, sample):
    model = kde.KernelDensity(kernel='gaussian', bandwidth=0.2).fit(sample)
    prob = np.exp(model.score_samples(vec.reshape(1, -1)))
    return prob


def pred_label(valid, train_sample1, train_sample2, train_sample3, method):
    label = []
    for i in range(valid.shape[0]):
        if method is 'likelihood':
            prob1 = cal_prob(valid[i, :], train_sample1)
            prob2 = cal_prob(valid[i, :], train_sample2)
            prob3 = cal_prob(valid[i, :], train_sample3)
        else:
            prob1 = cal_prob_smooth(valid[i, :], train_sample1)
            prob2 = cal_prob_smooth(valid[i, :], train_sample2)
            prob3 = cal_prob_smooth(valid[i, :], train_sample3)
        if max([prob1, prob2, prob3]) == prob1:  # 选择概率值最大的作为预测标签
            label.append(1)
        elif max([prob1, prob2, prob3]) == prob2:
            label.append(2)
        else:
            label.append(3)
    return np.array(label)


def cal_distance(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))  # 计算欧式距离


def get_knn(vec, data, k):
    distances = []  # 存储目标点与训练样本中的点的距离
    for i in range(data.shape[0]):  # 与训练样本中的每一个点都需要计算距离
        distance = cal_distance(vec, data[i])
        distances.append(distance)
    distances = np.array(distances)
    idx = np.argsort(distances)[0:k]  # 取出前k个距离最小的点对应的索引
    distances = distances[idx]  # 取出k个点的距离，其余点可以省略
    return distances, idx


def pred_label_knn(test, train, label, k):
    preds = []  # 存储对待分类点预测的标签
    for i in range(test.shape[0]):
        vec = test[i]
        vote = np.zeros(3)  # 投票数组，用于判定k个点中哪类点最多
        distances, idx = get_knn(vec, train, k)  # 计算出k个近邻
        knn_label = label[idx]  # 在训练样本标签中取出k个近邻的标签
        for j in range(knn_label.shape[0]):
            if knn_label[j] == 1:
                vote[0] += 1  # 进行投票
            elif knn_label[j] == 2:
                vote[1] += 1
            else:
                vote[2] += 1
        pred = vote.argmax() + 1
        preds.append(pred)
    return np.array(preds)


def pred_label_knn_weighted(test, train, label, k=3):
    preds = []  # 存储对待分类点预测的标签
    for i in range(test.shape[0]):
        vec = test[i]
        vote = np.zeros(3)  # 投票数组，用于判定k个点中哪类点最多
        distances, idx = get_knn(vec, train, k)  # 计算出k个近邻
        knn_label = label[idx]  # 在训练样本标签中取出k个近邻的标签
        for j in range(knn_label.shape[0]):
            if distances[j] == 0:
                continue
            else:
                if knn_label[j] == 1:
                    vote[0] += 1/distances[j]  # 基于距离加权的投票
                elif knn_label[j] == 2:
                    vote[1] += 1/distances[j]
                else:
                    vote[2] += 1/distances[j]
        pred = vote.argmax() + 1
        preds.append(pred)
    return np.array(preds)


def cross_valid(data, label, fold, method, k=3):
    acc = []
    for i in range(fold):
        train_sample1, train_sample2, train_sample3, \
        train_label1, train_label2, train_label3, \
        valid_sample, valid_label = train_valid_split(data, label, fold, i)
        if method is 'likelihood' or method is 'kernel':
            pred = pred_label(valid_sample, train_sample1, train_sample2,
                          train_sample3, method)
        elif method is 'knn':
            train_sample = np.concatenate((train_sample1, train_sample2, train_sample3),
                                          axis=0)
            train_label = np.concatenate((train_label1, train_label2, train_label3),
                                         axis=0)
            pred = pred_label_knn(valid_sample, train_sample, train_label, k)
        elif method is 'knn_weighted':
            train_sample = np.concatenate((train_sample1, train_sample2, train_sample3),
                                          axis=0)
            train_label = np.concatenate((train_label1, train_label2, train_label3),
                                         axis=0)
            pred = pred_label_knn_weighted(valid_sample, train_sample, train_label, k)
        else: return None
        correct = 0
        for j in range(valid_label.shape[0]):
            if pred[j] == valid_label[j]:
                correct += 1
        print(i, 'acc', method, correct/valid_label.shape[0])
        acc.append(correct/valid_label.shape[0])
    return np.mean(acc)


def select_k(data, label, fold, method):
    k = [x for x in range(1, 21)]
    accs = []
    for i in k:
        acc = cross_valid(data, label, fold, method, i)
        accs.append(acc)
    best_k = np.argmax(accs) + 1
    return best_k, k, accs


if __name__ == '__main__':
    data_path = 'HWData3.csv'
    fold = 10
    data, label = read_data(data_path)
    acc_likelihood = cross_valid(data, label, fold, method='likelihood')
    acc_kernel = cross_valid(data, label, fold, method='kernel')
    best_k, k, accs_knn = select_k(data, label, fold, method='knn')
    best_k_weighted, k, accs_knn_weighted = select_k(data, label, fold,
                                                     method='knn_weighted')
    acc_knn = cross_valid(data, label, fold, method='knn', k=best_k)
    acc_knn_weighted = cross_valid(data, label, fold, method='knn_weighted',
                                   k=best_k_weighted)
    print('acc using max likelihood estimation:', acc_likelihood)
    print('acc using smooth kernel function:', acc_kernel)
    print('acc using k nearest neighbour:', acc_knn)
    print('acc using weighted k nearest neighbour:', acc_knn_weighted)

    plt.title('K-NN Result Analysis')
    plt.plot(k, accs_knn, color='green', label='knn accuracy')
    plt.plot(k, accs_knn_weighted, color='red', label='weighted knn accuracy')
    plt.xticks(range(21))
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.show()