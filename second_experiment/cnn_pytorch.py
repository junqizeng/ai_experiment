import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.dataset_tensor_loader import ImgDataset
from torch.optim import Adam
from sklearn.metrics import accuracy_score

# from torchmetrics import Accuracy

"""

0.9947916666666666

"""


class ImgClassifier(nn.Module):
    def __init__(self):
        super(ImgClassifier, self).__init__()
        self.encoder = nn.Sequential(
            # 32
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),   # 30
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),  # 28
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2),  # 13
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),   # 11
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),   # 9
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2),  # 3
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),   # 1
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            # nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, img):
        img_fea = self.encoder(img)

        img_fea = img_fea.view(img_fea.size(0), -1)
        img_label = self.classifier(img_fea)
        return img_label


if __name__ == '__main__':
    train_set_path = './dataset/trainingDigits'
    test_set_path = './dataset/testDigits'
    device = 'cuda:0'

    train_set = ImgDataset(dataset_root_path=train_set_path)
    test_set = ImgDataset(dataset_root_path=test_set_path)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    img_classifier = ImgClassifier().to(device)
    optimizer = Adam(img_classifier.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()
    # acc_cal = Accuracy()

    max_epoch = 50
    max_acc = 0
    for epoch in range(max_epoch):
        print(epoch)
        print('---------------------------------------------')
        img_classifier.train()
        for data_batch, label_batch in train_loader:
            loss_list = []
            data_batch = data_batch.to(device)
            label_batch = label_batch.to(device)

            label_pred = img_classifier(data_batch)
            loss = loss_fn(label_pred, label_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_list.append(loss.item())
            # print(sum(loss_list)/len(loss_list))

        img_classifier.eval()
        acc_list = []
        test_num = 0
        for data_batch, label_batch in test_loader:
            test_num += len(label_batch)
            data_batch = data_batch.to(device)

            label_pred = img_classifier(data_batch).to('cpu')
            label_pred = torch.argmax(label_pred, dim=1)

            acc = accuracy_score(label_pred, label_batch)
            acc_list.append(acc)
            # acc = acc_cal(label_pred, label_batch)
            # acc_list.append(acc.item())
        print(f'{epoch} acc {sum(acc_list) / len(acc_list)}  {test_num}')
        max_acc = max(max_acc, sum(acc_list) / len(acc_list))

    print(max_acc)
