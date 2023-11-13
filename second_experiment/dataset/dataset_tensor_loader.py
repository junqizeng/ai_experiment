import torch
from torch.utils.data import Dataset
import os


class ImgDataset(Dataset):
    def __init__(self, dataset_root_path):
        super(Dataset, self).__init__()
        self.data_set = []
        self.label_set = []
        self.label_cnt_list = [0]*10

        for file in sorted(os.listdir(dataset_root_path)):
            img_label = int(file.split('_')[0])
            self.label_cnt_list[img_label] += 1
            self.label_set.append(img_label)
            with open(os.path.join(dataset_root_path, file), 'r') as f:
                sample = [[int(bit) for bit in seq] for seq in f.read().strip().split('\n')]
                self.data_set.append(sample)

    def __getitem__(self, idx):
        return torch.unsqueeze(torch.tensor(self.data_set[idx], dtype=torch.float32), 0), self.label_set[idx]

    def __len__(self):
        return len(self.data_set)


if __name__ == '__main__':
    path = './trainingDigits'
    dataset = ImgDataset(path)
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset, batch_size=5, shuffle=True)
    for datas, labels in data_loader:
        datas = datas.tolist()
        print(datas, labels)
