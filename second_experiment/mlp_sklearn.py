# 要求使用不同隐藏层节点数和学习率（node=500， 1000， 1500， 2000；lr=0.1， 0.01， 0.001， 0.0001） 错误数量、正确率
from sklearn.neural_network import MLPClassifier
from dataset.dataset_loader import Dataset_loader
from sklearn.metrics import accuracy_score

"""
500 0.1:0.9534883720930233 wrong_num:43.99999999999997
500 0.01:0.9799154334038055 wrong_num:19.000000000000004
500 0.001:0.9756871035940803 wrong_num:23.00000000000005
500 0.0001:0.9735729386892178 wrong_num:24.999999999999968
1000 0.1:0.9492600422832981 wrong_num:48.00000000000002
1000 0.01:0.9788583509513742 wrong_num:20.000000000000018
1000 0.001:0.9746300211416491 wrong_num:23.999999999999957
1000 0.0001:0.9778012684989429 wrong_num:21.00000000000003
1500 0.1:0.857293868921776 wrong_num:134.99999999999994
1500 0.01:0.9830866807610994 wrong_num:15.999999999999972
1500 0.001:0.9778012684989429 wrong_num:21.00000000000003
1500 0.0001:0.9799154334038055 wrong_num:19.000000000000004
2000 0.1:0.9471458773784355 wrong_num:50.00000000000004
2000 0.01:0.9820295983086681 wrong_num:16.999999999999982
2000 0.001:0.9809725158562368 wrong_num:17.999999999999993
2000 0.0001:0.9788583509513742 wrong_num:20.000000000000018
"""

if __name__ == "__main__":
    train_set_path = './dataset/trainingDigits'
    test_set_path = './dataset/testDigits'

    train_set = Dataset_loader(dataset_root_path=train_set_path)
    test_set = Dataset_loader(dataset_root_path=test_set_path)

    hidden_layer_list = [500, 1000, 1500, 2000]
    lr_list = [0.1, 0.01, 0.001, 0.0001]
    # hidden_layer_list = [1000]
    # lr_list = [0.01]
    max_epoch = 2000

    for hidden_layer in hidden_layer_list:
        for lr in lr_list:
            mlp_classifier = MLPClassifier(hidden_layer_sizes=hidden_layer, learning_rate_init=lr, max_iter=max_epoch,
                                           random_state=42)
            mlp_classifier.fit(train_set.data_set, train_set.label_set)

            label_pred = mlp_classifier.predict(test_set.data_set)
            acc = accuracy_score(y_true=test_set.label_set, y_pred=label_pred)
            print(f"{hidden_layer} {lr}:{acc} wrong_num:{946*(1-acc)}")
