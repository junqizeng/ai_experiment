import os


class Dataset_loader:
    def __init__(self, dataset_root_path):
        self.data_set = []
        self.label_set = []
        self.label_cnt_list = [0]*10

        for file in sorted(os.listdir(dataset_root_path)):
            img_label = int(file.split('_')[0])
            self.label_cnt_list[img_label] += 1
            self.label_set.append(img_label)
            with open(os.path.join(dataset_root_path, file), 'r') as f:
                sample = [int(bit) for bit in f.read().replace('\n', '')]
                self.data_set.append(sample)


if __name__ == '__main__':
    dataset = Dataset_loader(dataset_root_path='./testDigits')
