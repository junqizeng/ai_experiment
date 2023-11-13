# 要求使用不同隐藏层节点数和学习率（node=500， 1000， 1500， 2000；lr=0.1， 0.01， 0.001， 0.0001） 错误数量、正确率
# 最重要的是权重初始化的值不能大，另外要打乱样本，不然马上过拟合
import numpy as np
from dataset.dataset_loader import Dataset_loader
import random
import time
from tqdm import tqdm
from datetime import datetime
import sys

"""
500

  0%|          | 0/100 [00:00<?, ?it/s]epoch 0 loss 2.1157288397938023
epoch 0 acc 0.49577167019027485
  1%|          | 1/100 [00:10<16:40, 10.11s/it]epoch 1 loss 1.461005280655578
  2%|▏         | 2/100 [00:19<16:14,  9.94s/it]epoch 1 acc 0.7050739957716702
epoch 2 loss 1.1295172980605157
  3%|▎         | 3/100 [00:29<16:08,  9.98s/it]epoch 2 acc 0.7780126849894292
epoch 3 loss 0.9183798422905007
  4%|▍         | 4/100 [00:39<15:58,  9.98s/it]epoch 3 acc 0.7367864693446089
epoch 4 loss 0.8421371626096786
epoch 4 acc 0.8678646934460887

0.9069767441860465
"""


def get_time():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return current_time


def print_with_time(content):
    print(f'{get_time()} {content}')
    sys.stdout.flush()
    return


class Optimizer:
    def __init__(self, lr, param_dict, gradient_dict):
        self.learning_rate = lr
        self.param_dict = param_dict
        self.gradient_dict = gradient_dict

    def optimize(self):
        for param_name in self.gradient_dict.keys():
            layer_gradient = self.gradient_dict[param_name]
            layer_weights = self.param_dict[param_name]
            self.param_dict[param_name] = layer_weights - self.learning_rate * layer_gradient
        return


def initialize_weights(rows, cols):
    return np.random.randn(rows, cols) * np.sqrt(2 / (rows + cols))


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.param_dict = {
            'hidden_layer_weights': initialize_weights(self.hidden_dim, self.input_dim),
            'hidden_layer_bias': initialize_weights(self.hidden_dim, 1),
            'output_layer_weights': initialize_weights(self.output_dim, self.hidden_dim),
            'output_layer_bias': initialize_weights(self.output_dim, 1)
        }

        self.gradient_dict = {
            'hidden_layer_weights': np.zeros((self.hidden_dim, self.input_dim)),
            'hidden_layer_bias': np.zeros((self.hidden_dim, 1)),
            'output_layer_weights': np.zeros((self.output_dim, self.hidden_dim)),
            'output_layer_bias': np.zeros((self.output_dim, 1))
        }

        self.x0 = None
        self.x1 = None
        self.out1 = None
        self.x2 = None
        self.out2 = None

        self.loss = None
        self.label = None
        self.smooth_term = 1e-8

    def loss_calculate(self, y_pred, y_label_):
        # y_pred = [10]   y_label_ = int
        y_label = np.zeros_like(y_pred)
        y_label[y_label_] = 1
        self.label = y_label
        # y_pred = [10] y_label = [10]
        y_pred = np.log(y_pred)
        self.loss = - y_pred * y_label
        return np.sum(self.loss)

    def softmax(self, x):
        # x [output, 1]
        x_exp = np.exp(x - np.max(x))
        return x_exp / np.sum(x_exp)

    def forward(self, input_x):
        # input_x = [1024]
        self.x0 = np.array([input_x]).T         # [1024, 1]
        self.x1 = self.param_dict['hidden_layer_weights'] @ self.x0 + self.param_dict['hidden_layer_bias']  # [hidden, 1]
        self.out1 = np.maximum(0, self.x1)      # relu [hidden, 1]

        self.x2 = self.param_dict['output_layer_weights'] @ self.out1 + self.param_dict['output_layer_bias'] # [output, 1]
        self.out2 = self.softmax(self.x2)  # [output, 1]
        return self.out2  # [output, 1]

    def backward(self):
        g_loss_out2 = - self.label / (self.out2 + self.smooth_term)
        g_out2_x2 = self.out2 * (1 - self.out2)
        g_x2_w2 = self.out1
        g = (g_loss_out2 * g_out2_x2) @ g_x2_w2.T
        self.gradient_dict['output_layer_weights'] = g  # [output, hidden]
        g = g_loss_out2 * g_out2_x2
        self.gradient_dict['output_layer_bias'] = g

        g_x2_out1 = self.param_dict['output_layer_weights']
        g_out1_x1 = np.where(self.out1 != 0, 1, 0)
        g_x1_w1 = self.x0
        g = ((g_x2_out1.T @ (g_loss_out2 * g_out2_x2)) * g_out1_x1) @ g_x1_w1.T
        self.gradient_dict['hidden_layer_weights'] = g
        g = ((g_x2_out1.T @ (g_loss_out2 * g_out2_x2)) * g_out1_x1)
        self.gradient_dict['hidden_layer_bias'] = g
        return

    def gradient_clear(self):
        # 该函数将保存的梯度置零
        for key in self.gradient_dict.keys():
            self.gradient_dict[key] = np.zeros_like(self.gradient_dict[key])


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    hidden_dim = 1000
    model = MLP(input_dim=1024, hidden_dim=hidden_dim, output_dim=10)
    optimizer = Optimizer(lr=0.0001, param_dict=model.param_dict, gradient_dict=model.gradient_dict)
    optimizer.optimize()
    result_save_path = f'./result/{get_time()}_{hidden_dim}.out'

    result = model.softmax(np.array([1, 10]))

    train_set_path = './dataset/trainingDigits'
    test_set_path = './dataset/testDigits'

    train_set = Dataset_loader(dataset_root_path=train_set_path)
    test_set = Dataset_loader(dataset_root_path=test_set_path)

    max_epoch = 100

    for i in range(max_epoch):
        train_data = train_set.data_set
        train_data_label = train_set.label_set
        # print(len(train_data))
        train_id_list = list(range(len(train_data)))
        loss_list = []
        random.shuffle(train_id_list)
        # train
        for j in train_id_list:
            input_sample = np.array(train_data[j])
            label = train_data_label[j]

            label_pred = model.forward(input_sample)
            loss = model.loss_calculate(y_label_=label, y_pred=label_pred)
            # print(loss, label)
            model.backward()
            optimizer.optimize()
            model.gradient_clear()
            # time.sleep(0.1)
            loss_list.append(loss)
        train_result_str = f"epoch {i} loss {sum(loss_list) / len(loss_list)}"
        print_with_time(train_result_str)

        optimizer.learning_rate *= 0.92

        # # test
        acc_list = []
        for j in range(len(test_set.data_set)):
            input_sample = test_set.data_set[j]
            label = test_set.label_set[j]
            label_pred = np.argmax(np.ravel(model.forward(input_sample)))
            acc_list.append(1 if label == label_pred else 0)
        test_result_str = f"epoch {i} acc {sum(acc_list)/len(acc_list)}"
        print_with_time(test_result_str)

        with open(result_save_path, 'a') as f:
            f.write(train_result_str+'\n')
            f.write(test_result_str+'\n')
    # print()

