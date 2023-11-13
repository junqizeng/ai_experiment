# 要求使用不同隐藏层节点数和学习率（node=500， 1000， 1500， 2000；lr=0.1， 0.01， 0.001， 0.0001） 错误数量、正确率
# 最重要的是权重初始化的值不能大，另外要打乱样本，不然马上过拟合
import numpy as np
from dataset.dataset_loader import Dataset_loader
import random
import time


class Optimizer:
    def __init__(self, lr, param_dict, gradient_dict):
        self.learning_rate = lr
        self.param_dict = param_dict
        self.gradient_dict = gradient_dict

    def optimize(self):
        for layer_name in self.gradient_dict.keys():
            layer_gradient = self.gradient_dict[layer_name]
            layer_weights = self.param_dict[layer_name + '_weights']
            self.param_dict[layer_name + '_weights'] = layer_weights - self.learning_rate * layer_gradient
        return


class MLP:
    def __init__(self):
        self.param_dict = {
            'hidden_layer_weights': np.array([[1, -1], [-2, 1]]),
            'hidden_layer_bias': np.array([[1], [1]]),
            'output_layer_weights': np.array([[2, -2], [-2, -1]]),
            'output_layer_bias': np.array([[1], [1]])
        }
        self.gradient_dict = {
            'hidden_layer': np.zeros((2, 2)),
            'output_layer': np.zeros((2, 2))
        }

        self.x0 = None
        self.x1 = None
        self.out1 = None
        self.x2 = None
        self.out2 = None

        self.loss = None
        self.label = None
        self.smooth_term = 1e-5

    def loss_calculate(self, y_pred, y_label_):
        # y_pred = [10]   y_label_ = int
        y_label = np.zeros_like(y_pred)
        y_label[y_label_] = 1
        self.label = y_label
        # y_pred = [10] y_label = [10]
        y_pred = np.log(y_pred + self.smooth_term)
        self.loss = - y_pred * y_label
        return np.sum(self.loss)

    def softmax(self, x):
        # x [output, 1]
        x_exp = np.exp(x - np.max(x))
        return x_exp / np.sum(x_exp)

    def forward(self, input_x):
        # input_x = [1024]
        self.x0 = np.array([input_x]).T  # [1024, 1]
        self.x1 = self.param_dict['hidden_layer_weights'] @ self.x0 + self.param_dict[
            'hidden_layer_bias']  # [hidden, 1]
        self.out1 = np.maximum(0, self.x1)  # relu [hidden, 1]

        self.x2 = self.param_dict['output_layer_weights'] @ self.out1 + self.param_dict[
            'output_layer_bias']  # [output, 1]
        self.out2 = np.ravel(self.softmax(self.x2))  # [output]
        return self.out2  # [output]

    def backward(self):
        g_loss_out2 = - self.label / (self.out2 + self.smooth_term)
        g_out2_x2 = self.out2 * (1 - self.out2)
        g_x2_w2 = self.out1
        g = (g_x2_w2 @ np.array([g_loss_out2 * g_out2_x2])).T
        self.gradient_dict['output_layer'] = g  # [output, hidden]
        g_x2_out1 = self.param_dict['output_layer_weights']
        g_out1_x1 = np.where(self.out1 != 0, 1, 0)
        g_x1_w1 = self.x0
        g = (g_x1_w1 @ ((np.array([g_loss_out2 * g_out2_x2]) @ g_x2_out1) * g_out1_x1.T)).T
        self.gradient_dict['hidden_layer'] = g
        return

    def gradient_clear(self):
        # 该函数将保存的梯度置零
        for key in self.gradient_dict.keys():
            self.gradient_dict[key] = np.zeros_like(self.gradient_dict[key])


if __name__ == '__main__':

    model = MLP()
    optimizer = Optimizer(lr=0.5, param_dict=model.param_dict, gradient_dict=model.gradient_dict)
    optimizer.optimize()

    input_sample = np.array([1, -1])
    label = 1
    for i in range(20):
        label_pred = model.forward(input_sample)
        loss = model.loss_calculate(y_pred=label_pred, y_label_=label)
        model.backward()
        optimizer.optimize()
        model.gradient_clear()
        print(loss)
    # train_set_path = './dataset/trainingDigits'
    # test_set_path = './dataset/testDigits'

    # train_set = Dataset_loader(dataset_root_path=train_set_path)
    # test_set = Dataset_loader(dataset_root_path=test_set_path)

    # for i in range(50):
    #     train_data = train_set.data_set
    #     train_data_label = train_set.label_set
    #     # print(len(train_data))
    #     train_id_list = list(range(len(train_data)))
    #     loss_list = []
    #     random.shuffle(train_id_list)
    #     # train
    #     for j in train_id_list:
    #         input_sample = np.array(train_data[j])
    #         label = train_data_label[j]
    #
    #         label_pred = model.forward(input_sample)
    #         loss = model.loss_calculate(y_label_=label, y_pred=label_pred)
    #         print(loss)
    #         model.backward()
    #         optimizer.optimize()
    #         model.gradient_clear()
    #         time.sleep(0.05)
    #         loss_list.append(loss)
    # print(sum(loss_list) / len(loss_list))

    # # test
    # for j in range(len(test_set.data_set)):
    #     input_sample = test_set.data_set[j]

    print()
