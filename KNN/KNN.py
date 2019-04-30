import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def read_data(path):
    raw = np.loadtxt(path)
    data = raw[:, 0:-10]  # 倒数十列之前全为训练样本
    label = raw[:, -10:]  # 倒数十列为标签
    return data, label


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


def predict_label(test, train, label, k):
    preds = []  # 存储对待分类点预测的标签
    for i in range(test.shape[0]):
        vec = test[i]
        vote = np.zeros(10)  # 投票数组，用于判定k个点中哪类点最多
        pred = list(np.zeros(10))  # 标签为one-hot形式，所以声明一个10×1的全零数组
        distances, idx = get_knn(vec, train, k)  # 计算出k个近邻
        knn_label = label[idx]  # 在训练样本标签中取出k个近邻的标签
        for j in range(knn_label.shape[0]):
            vote[np.nonzero(knn_label[j])] += 1  # 进行投票
        pred[vote.argmax()] = 1
        preds.append(pred)
    return np.array(preds)


def predict_label_weighted(test, train, label, k):
    preds = []  # 存储对待分类点预测的标签
    for i in range(test.shape[0]):
        vec = test[i]
        vote = np.zeros(10)  # 投票数组，用于判定k个点中哪类点最多
        pred = list(np.zeros(10))  # 标签为one-hot形式，所以声明一个10×1的全零数组
        distances, idx = get_knn(vec, train, k)  # 计算出k个近邻
        knn_label = label[idx]  # 在训练样本标签中取出k个近邻的标签
        for j in range(knn_label.shape[0]):
            if distances[j] == 0:
                vote[np.nonzero(knn_label[j])] += 0
            else:
                vote[np.nonzero(knn_label[j])] += 1/distances[j]  # 基于距离加权的投票
        pred[vote.argmax()] = 1
        preds.append(pred)
    return np.array(preds)


def train_valid_split(data, label, fold, idx):
    data = list(data)
    label = list(label)
    train_data = []
    train_label = []
    valid_data = []
    valid_label = []
    for i in range(data.__len__()):
        if i%fold == idx:  # 采用取模的方式来进行验证集的划分
            valid_data.append(data[i])
            valid_label.append(label[i])
        else:
            train_data.append(data[i])
            train_label.append(label[i])
    return np.array(train_data), np.array(train_label), \
           np.array(valid_data), np.array(valid_label)


def cross_valid(data, label, fold, k):
    losses = []
    for i in range(fold):  # 进行多折交叉验证
        train_data, train_label, \
        valid_data, valid_label = train_valid_split(data, label, fold, i)
        pred_label = predict_label(valid_data, train_data, train_label, k)
        loss = 1 - metrics.accuracy_score(valid_label, pred_label)
        losses.append(loss)
    return np.mean(losses)  # 返回交叉验证计算出的平均误差


def select_k(data, label):
    losses = []
    for i in range(1, 21):  # 所选择的k值范围为1-20
        loss = cross_valid(data, label, 10, i)  # 进行10折交叉验证
        losses.append(loss)
        print(i, "NN loss is ", loss)
    return np.argmin(losses) + 1, losses  # 返回loss值最小的k值


def cross_valid_weighted(data, label, fold, k):
    losses = []
    for i in range(fold):  # 进行多折交叉验证
        train_data, train_label, \
        valid_data, valid_label = train_valid_split(data, label, fold, i)
        pred_label = predict_label_weighted(valid_data, train_data, train_label, k)
        loss = 1 - metrics.accuracy_score(valid_label, pred_label)
        losses.append(loss)
    return np.mean(losses)  # 返回交叉验证计算出的平均误差


def select_k_weighted(data, label):
    losses = []
    for i in range(1, 21):  # 所选择的k值范围为1-20
        loss = cross_valid_weighted(data, label, 10, i)  # 进行10折交叉验证
        losses.append(loss)
        print(i, "NN distance-weighted loss is ", loss)
    return np.argmin(losses) + 1, losses  # 返回loss值最小的k值


if __name__ == '__main__':
    train_path = 'semeion_train.csv'
    test_path = 'semeion_test.csv'
    train_data, train_label = read_data(train_path)
    test_data, test_label = read_data(test_path)

    best_k, losses = select_k(train_data, train_label)
    print("The best value of k based on simple "
          "K-NN from 10-fold cross validation is", best_k)
    best_k_weighted, losses_weighted = select_k_weighted(train_data, train_label)
    print("The best value of k based on distance-weighted "
          "K-NN from 10-fold cross validation is", best_k_weighted)

    pred_label = predict_label(test_data, train_data, train_label, best_k)
    loss = 1 - metrics.accuracy_score(test_label, pred_label)
    pred_weighted = predict_label_weighted(test_data, train_data, train_label, best_k_weighted)
    loss_weighted = 1 - metrics.accuracy_score(test_label, pred_weighted)
    print("my simple knn loss:", loss)
    print("my distance-weighted knn loss:", loss_weighted)

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(train_data, train_label)
    pred_sklearn = knn.predict(test_data)
    loss_sklearn = 1 - metrics.accuracy_score(test_label, pred_sklearn)
    print("sklearn knn loss: ", loss_sklearn)

    losses_sklearn = []
    for i in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(train_data, train_label)
        pred_sklearn = knn.predict(test_data)
        loss_sklearn = 1 - metrics.accuracy_score(test_label, pred_sklearn)
        losses_sklearn.append(loss_sklearn)

    k = [x for x in range(1, 21)]

    plt.title('Result Analysis')
    plt.plot(k, losses, color='green', label='simple knn loss')
    plt.plot(k, losses_sklearn, color='red', label='sklearn knn loss')
    plt.plot(k, losses_weighted, color='blue', label='distance-weighted knn loss')
    plt.legend()  # 显示图例

    plt.xlabel('k')
    plt.ylabel('loss')
    plt.show()