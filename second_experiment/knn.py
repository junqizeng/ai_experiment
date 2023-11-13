# 要求使用不同邻居数量做对比（k=1，3，5，7）

"""
训练数据情况：
[189, 198, 195, 199, 186, 187, 195, 201, 180, 204]
1934
测试数据情况：
[87, 97, 92, 85, 114, 108, 87, 96, 91, 89]
946
1: 0.9862579281183932
3: 0.9873150105708245
5: 0.9809725158562368
7: 0.9767441860465116
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from dataset.dataset_loader import Dataset_loader

if __name__ == '__main__':
    train_set_path = './dataset/trainingDigits'
    test_set_path = './dataset/testDigits'

    train_set = Dataset_loader(dataset_root_path=train_set_path)
    test_set = Dataset_loader(dataset_root_path=test_set_path)

    print("训练数据情况：")
    print(train_set.label_cnt_list)
    print(sum(train_set.label_cnt_list))
    print("测试数据情况：")
    print(test_set.label_cnt_list)
    print(sum(test_set.label_cnt_list))

    k_list = [1, 3, 5, 7]

    for k in k_list:
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(train_set.data_set, train_set.label_set)

        label_pred = knn_classifier.predict(test_set.data_set)
        acc = accuracy_score(y_true=test_set.label_set, y_pred=label_pred)
        print(f'{k}: {acc}')
