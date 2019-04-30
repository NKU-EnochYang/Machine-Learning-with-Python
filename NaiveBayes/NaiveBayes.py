import numpy as np
import random
import matplotlib.pyplot as plt


# 读取数据
def read_data(path):
    data = list(np.loadtxt(path, delimiter=','))
    # 打乱数据，为方便训练集和测试集划分
    random.shuffle(data)
    data = np.array(data)
    sample1 = []
    sample2 = []
    sample3 = []
    for each in data:
        if each[0] == 1:
            sample1.append(each)
        elif each[0] == 2:
            sample2.append(each)
        else:
            sample3.append(each)
    return np.array(sample1), np.array(sample2), np.array(sample3)


def train_test_split(sample1, sample2, sample3, fold, idx):
    train_sample1 = []
    train_sample2 = []
    train_sample3 = []
    test_sample = []
    for i in range(sample1.shape[0]):
        if i % fold != idx:
            train_sample1.append(sample1[i])
        else:
            test_sample.append(sample1[i])
    for i in range(sample2.shape[0]):
        if i % fold != idx:
            train_sample2.append(sample2[i])
        else:
            test_sample.append(sample2[i])
    for i in range(sample3.shape[0]):
        if i % fold != idx:
            train_sample3.append(sample3[i])
        else:
            test_sample.append(sample3[i])
    train_sample1 = np.array(train_sample1)
    train_sample2 = np.array(train_sample2)
    train_sample3 = np.array(train_sample3)
    train_label1 = train_sample1[:, 0]
    train_label2 = train_sample2[:, 0]
    train_label3 = train_sample3[:, 0]
    train_sample1 = train_sample1[:, 1:]
    train_sample2 = train_sample2[:, 1:]
    train_sample3 = train_sample3[:, 1:]
    test_sample = np.array(test_sample)
    test_label = test_sample[:, 0]
    test_sample = test_sample[:, 1:]
    return train_sample1, train_label1, train_sample2, train_label2, \
           train_sample3, train_label3, test_sample, test_label


def cal_para(sample):
    mean = np.mean(sample, 0)
    var = np.var(sample, 0)
    return mean, var


def cal_prob(test_vec, train_sample, total):
    mean, var = cal_para(train_sample)
    # 根据高斯函数计算
    prob = (1 / np.sqrt(2*np.pi*var)) * np.exp(-np.square(test_vec - mean) / (2*var))
    # 先验概率
    prior = train_sample.shape[0] / total
    likelihood = 1
    for each in prob:
        # 连乘条件概率
        likelihood *= each
    return prior*likelihood


def decide_label(test_sample, train_sample1, train_sample2, train_sample3):
    total = train_sample1.shape[0] + train_sample2.shape[0] + train_sample3.shape[0]
    label = []
    score = []
    for each in test_sample:
        prob1 = cal_prob(each, train_sample1, total)
        prob2 = cal_prob(each, train_sample2, total)
        prob3 = cal_prob(each, train_sample3, total)
        # 选择概率最大的作为预测标签
        if max([prob1, prob2, prob3]) == prob1:
            label.append(1)
        elif max([prob1, prob2, prob3]) == prob2:
            label.append(2)
        else:
            label.append(3)
        score.append(prob1)
    return np.array(label), np.array(score)


# 交叉验证
def cross_valid(sample1, sample2, sample3, fold):
    acc = []
    for i in range(fold):
        train_sample1, train_label1, train_sample2, train_label2, \
        train_sample3, train_label3, \
        test_sample, test_label = train_test_split(sample1, sample2, sample3, fold, i)
        pred_label, pred_score = decide_label(test_sample, train_sample1, train_sample2, train_sample3)
        correct = 0
        for j in range(test_label.shape[0]):
            if pred_label[j] == test_label[j]:
                correct += 1
        print(i, 'acc:', correct / test_label.shape[0])
        acc.append(correct / test_label.shape[0])
    return np.mean(acc)


def result_analysis(sample1, sample2, sample3):
    train_sample1, train_label1, train_sample2, train_label2, \
    train_sample3, train_label3, \
    test_sample, test_label = train_test_split(sample1, sample2, sample3, fold, 1)
    pred_label, pred_score = decide_label(test_sample, train_sample1, train_sample2, train_sample3)
    cm = np.zeros((3, 3))
    # 构造混淆矩阵
    for i in range(test_label.shape[0]):
        cm[int(test_label[i]) - 1, pred_label[i] - 1] += 1
    tp1 = cm[0, 0]
    tp2 = cm[1, 1]
    tp3 = cm[2, 2]
    fn1 = cm[0, 1] + cm[0, 2]
    fn2 = cm[1, 0] + cm[1, 2]
    fn3 = cm[2, 0] + cm[2, 1]
    fp1 = cm[1, 0] + cm[2, 0]
    fp2 = cm[0, 1] + cm[2, 1]
    fp3 = cm[0, 2] + cm[1, 2]
    tn1 = cm[1, 1] + cm[1, 2] + cm[2, 1] + cm[2, 2]
    tn2 = cm[0, 0] + cm[0, 2] + cm[2, 0] + cm[2, 2]
    tn3 = cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1]
    # 根据公式计算precision、recall和F1
    precision1 = tp1 / (tp1 + fp1)
    precision2 = tp2 / (tp2 + fp2)
    precision3 = tp3 / (tp3 + fp3)
    recall1 = tp1 / (tp1 + fn1)
    recall2 = tp2 / (tp2 + fn2)
    recall3 = tp3 / (tp3 + fn3)
    f1 = (2*precision1*recall1) / (precision1 + recall1)
    f2 = (2*precision2*recall2) / (precision2 + recall2)
    f3 = (2*precision3*recall3) / (precision3 + recall3)
    return cm, precision1, precision2, precision3, recall1, recall2, recall3, f1, f2, f3


def draw_roc(sample1, sample2, sample3):
    train_sample1, train_label1, train_sample2, train_label2, \
    train_sample3, train_label3, \
    test_sample, test_label = train_test_split(sample1, sample2, sample3, fold, 1)
    pred_label, pred_score = decide_label(test_sample, train_sample1, train_sample2, train_sample3)
    fpr = []
    tpr = []
    # 计算fpr和tpr，阈值从大到小
    for i in np.argsort(pred_score)[::-1]:
        threshold = pred_score[i]
        if threshold == np.max(pred_score):
            fpr.append(0)
            tpr.append(0)
            continue
        elif threshold == np.min(pred_score):
            fpr.append(1)
            tpr.append(1)
            continue
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for j in np.argsort(pred_score)[::-1]:
            if pred_score[j] >= threshold:
                if test_label[j] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if test_label[j] == 1:
                    fn += 1
                else:
                    tn += 1
        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))
    auc = 0
    for i in range(1, len(fpr)):
        auc += 0.5 * (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1])
    plt.title('ROC')
    plt.plot(fpr, tpr, color='green', label='ROC')
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.show()
    return auc


if __name__ == '__main__':
    data_path = 'wine.data'
    fold = 5
    sample1, sample2, sample3 = read_data(data_path)
    acc = cross_valid(sample1, sample2, sample3, fold)
    print(fold, 'fold cross validation mean acc:', acc)
    cm, precision1, precision2, precision3, recall1, recall2, recall3, \
    f1, f2, f3 = result_analysis(sample1, sample2, sample3)
    print('Result analysis of the second fold cross validation')
    print('Confusion matrix:')
    print(cm)
    print('precision for 1:', precision1)
    print('precision for 2:', precision2)
    print('precision for 3:', precision3)
    print('recall for 1:', recall1)
    print('recall for 2:', recall2)
    print('recall for 3:', recall3)
    print('F1-score for 1:', f1)
    print('F1-score for 2:', f2)
    print('F1-score for 3:', f3)
    auc = draw_roc(sample1, sample2, sample3)
    print('AUC:', auc)