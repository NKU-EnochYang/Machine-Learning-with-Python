import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from itertools import permutations
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

def create_data(centers,num=100,std=0.7):
    '''
    生成用于聚类的数据集
    :param centers: 聚类的中心点组成的数组。如果中心点是二维的，则产生的每个样本都是二维的。
    :param num: 样本数
    :param std: 每个簇中样本的标准差
    :return: 用于聚类的数据集。是一个元组，第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    '''
    X, labels_true = make_blobs(n_samples=num, centers=centers, cluster_std=std)
    return  X,labels_true

def plot_data(*data, method='single linkage'):
    '''
    绘制用于聚类的数据集
    :param data: 可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X,labels_true=data
    labels=np.unique(labels_true)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    colors='rgbyckm' # 每个簇的样本标记不同的颜色
    for i,label in enumerate(labels):
        position=labels_true==label
        ax.scatter(X[position,0],X[position,1],label="cluster %d"%label,
		color=colors[i%len(colors)])

    ax.legend(loc="best",framealpha=0.5)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[1]")
    ax.set_title(method)
    plt.show()

def cal_node_distance(node1, node2):
    return np.sqrt(np.sum(np.square(node1 - node2)))  # 计算欧式距离

def cal_set_distance(set1, set2, method='average'):
    '''计算两个cluster间的距离，将其当作数据点的集合，遍历地进行处理'''
    dists = []
    for i in range(set1.shape[0]):
        cur1 = set1[i]
        for j in range(set2.shape[0]):
            cur2 = set2[j]
            dist = cal_node_distance(cur1, cur2)
            dists.append(dist)
    if method is 'single':
        return np.min(dists)
    elif method is 'complete':
        return np.max(dists)
    elif method is 'average':
        return np.mean(dists)
    else:
        raise IOError('Illegal method')

def single_linkage_clustering(sample, k):
    clusters = []
    for i in range(sample.shape[0]):
        node = []
        node.append(i)
        clusters.append(node)
    '''计算距离矩阵'''
    dists_matrix = []
    for i in range(len(clusters)):
        dists = []
        cur_set1 = sample[clusters[i]]
        for j in range(len(clusters)):
            cur_set2 = sample[clusters[j]]
            dist = cal_set_distance(cur_set1, cur_set2, method='single')
            dists.append(dist)
        dists_matrix.append(dists)
    dists_matrix = np.array(dists_matrix)
    '''直至该层次下只剩我们想要的k个簇'''
    while len(clusters) != k:
        base_idx = np.argmin(np.min(dists_matrix, 1))
        base_dist = dists_matrix[base_idx]
        target_idx = np.argsort(base_dist)[1] # 选取除本身外最小距离的cluster进行聚合
        for each in clusters[target_idx]:
            clusters[base_idx].append(each)
        del clusters[target_idx]
        dists_matrix = np.delete(dists_matrix, target_idx, axis=0)
        dists_matrix = np.delete(dists_matrix, target_idx, axis=1)
        if base_idx > target_idx:
            base_idx -= 1 # 如果被聚合cluster索引在基cluster之前，删除之后需将基cluster索引减1
        for i in range(len(clusters)):
            dists_matrix[base_idx, i] = cal_set_distance(sample[clusters[base_idx]],
                                                         sample[clusters[i]], method='single')
            dists_matrix[i, base_idx] = dists_matrix[base_idx, i] # 更新距离矩阵
        # if len(clusters) % 10 == 0:
        print('single linkage clustering number of clusters:', len(clusters))
    return clusters, dists_matrix

def complete_linkage_clustering(sample, k):
    '''思路与single linkage一样，将距离计算参数修改即可'''
    clusters = []
    for i in range(sample.shape[0]):
        node = []
        node.append(i)
        clusters.append(node)
    dists_matrix = []
    for i in range(len(clusters)):
        dists = []
        cur_set1 = sample[clusters[i]]
        for j in range(len(clusters)):
            cur_set2 = sample[clusters[j]]
            dist = cal_set_distance(cur_set1, cur_set2, method='complete')
            dists.append(dist)
        dists_matrix.append(dists)
    dists_matrix = np.array(dists_matrix)
    while len(clusters) != k:
        base_idx = np.argmin(np.min(dists_matrix, 1))
        base_dist = dists_matrix[base_idx]
        target_idx = np.argsort(base_dist)[1]
        for each in clusters[target_idx]:
            clusters[base_idx].append(each)
        del clusters[target_idx]
        dists_matrix = np.delete(dists_matrix, target_idx, axis=0)
        dists_matrix = np.delete(dists_matrix, target_idx, axis=1)
        if base_idx > target_idx:
            base_idx -= 1
        for i in range(len(clusters)):
            dists_matrix[base_idx, i] = cal_set_distance(sample[clusters[base_idx]],
                                                         sample[clusters[i]], method='complete')
            dists_matrix[i, base_idx] = dists_matrix[base_idx, i]
        # if len(clusters) % 10 == 0:
        print('complete linkage clustering number of clusters:', len(clusters))
    return clusters, dists_matrix

def average_linkage_clustering(sample, k):
    '''与前两种算法思路一致，修改距离计算方式即可'''
    clusters = []
    for i in range(sample.shape[0]):
        node = []
        node.append(i)
        clusters.append(node)
    dists_matrix = []
    for i in range(len(clusters)):
        dists = []
        cur_set1 = sample[clusters[i]]
        for j in range(len(clusters)):
            cur_set2 = sample[clusters[j]]
            dist = cal_set_distance(cur_set1, cur_set2, method='average')
            dists.append(dist)
        dists_matrix.append(dists)
    dists_matrix = np.array(dists_matrix)
    while len(clusters) != k:
        base_idx = np.argmin(np.min(dists_matrix, 1))
        base_dist = dists_matrix[base_idx]
        target_idx = np.argsort(base_dist)[1]
        for each in clusters[target_idx]:
            clusters[base_idx].append(each)
        del clusters[target_idx]
        dists_matrix = np.delete(dists_matrix, target_idx, axis=0)
        dists_matrix = np.delete(dists_matrix, target_idx, axis=1)
        if base_idx > target_idx:
            base_idx -= 1
        for i in range(len(clusters)):
            dists_matrix[base_idx, i] = cal_set_distance(sample[clusters[base_idx]],
                                                         sample[clusters[i]], method='average')
            dists_matrix[i, base_idx] = dists_matrix[base_idx, i]
        # if len(clusters) % 10 == 0:
        print('average linkage clustering number of clusters:', len(clusters))
    return clusters, dists_matrix

def cal_acc(pred, true):
    correct = 0
    for i in range(pred.shape[0]):
        if pred[i] == true[i]:
            correct += 1
    return correct/pred.shape[0]

def result_analysis(sample, label, k, method='average'):
    '''结果分析接口'''
    if method is 'single':
        clusters, dists_matrix = single_linkage_clustering(sample, k)
    elif method is 'average':
        clusters, dists_matrix = average_linkage_clustering(sample, k)
    elif method is 'complete':
        clusters, dists_matrix = complete_linkage_clustering(sample, k)
    accs = []
    pred_labels = []
    cases = [x for x in range(k)]
    cases = permutations(cases) # 0、1、2、3四种标签的排列情况
    for case in cases:
        pred_label = np.zeros(label.shape)
        for i in range(case.__len__()):
            pred_label[clusters[i]] = case[i]
        pred_labels.append(pred_label)
        acc = cal_acc(pred_label, label)
        accs.append(acc)
    acc = np.max(accs) # 将排列中准确度最高的情况作为预测标签
    pred_label = pred_labels[np.argmax(accs)]
    return acc, pred_label

def perform_test():
    '''测试簇心数量对模型性能的影响'''
    accs_single = []
    accs_complete = []
    accs_average = []
    samples = []
    centers = [[1, 1, 1], [1, 3, 3], [3, 6, 5], [2, 6, 8], [3, 2, 2],
               [1, 5, 2], [3, 1, 4], [4, 2, 1], [2, 3, 1], [3, 3, 5]]
    centers = np.array(centers)
    labels_true = []
    labels_single = []
    labels_complete = []
    labels_average = []
    for i in range(2, 8):
        center = centers[[x for x in range(i)]]
        X, label_true = create_data(center, 300, 0.5)  # 产生用于聚类的数据集，聚类中心点的个数代表类别数
        acc_single, label_single = result_analysis(X, label_true, i, method='single')
        acc_complete, label_complete = result_analysis(X, label_true, i, method='complete')
        acc_average, label_average = result_analysis(X, label_true, i, method='average')
        accs_single.append(acc_single)
        accs_complete.append(acc_complete)
        accs_average.append(acc_average)
        samples.append(X)
        labels_true.append(label_true)
        labels_single.append(label_single)
        labels_complete.append(label_complete)
        labels_average.append(label_average)
    return samples, labels_true, labels_single, labels_complete, labels_average, \
           accs_single, accs_complete, accs_average

if __name__=='__main__':
    centers = [[1,1,1], [1,3,3], [3,6,5], [2,6,8]] # 用于产生聚类的中心点, 聚类中心的维度代表产生样本的维度
    X,labels_true = create_data(centers, 1000, 0.5) # 产生用于聚类的数据集，聚类中心点的个数代表类别数
    print(X.shape)
    plot_data(X, labels_true, method='True')
    acc_single, labels_single = result_analysis(X, labels_true, 4, method='single')
    acc_complete, labels_complete = result_analysis(X, labels_true, 4, method='complete')
    acc_average, labels_average = result_analysis(X, labels_true, 4, method='average')
    print('acc using single linkage:', acc_single)
    print('acc using complete linkage:', acc_complete)
    print('acc using average linkage:', acc_average)
    plot_data(X, labels_single, method='Single linkage')
    plot_data(X, labels_complete, method='complete linkage')
    plot_data(X, labels_average, method='Average linkage')

    test_samples, test_labels_true, test_labels_single, test_labels_complete, \
    test_labels_average, test_accs_single, test_accs_complete, test_accs_average = perform_test()
    k = [x for x in range(2, 8)]

    plt.title('Result Analysis')
    plt.plot(k, test_accs_single, color='green', label='single linkage acc')
    plt.plot(k, test_accs_complete, color='red', label='complete linkage acc')
    plt.plot(k, test_accs_average, color='blue', label='average linkage acc')

    plt.legend()  # 显示图例

    plt.xlabel('Number of centers of clusters')
    plt.ylabel('accuracy')
    plt.show()