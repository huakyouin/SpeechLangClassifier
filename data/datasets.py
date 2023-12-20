
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 定义一个MelFrequencyDataset数据集类
class MelFrequencyDataset(Dataset):
    def __init__(self, data_dir, max_seq_length=300):
        self.data = []
        self.labels = []
        self.max_seq_length = max_seq_length
        label_folders = os.listdir(data_dir)
        for label_folder in label_folders:
            label = 0 if label_folder=='language_0' else 1  # 文件夹名字即为标签
            record_path = os.path.join(data_dir, label_folder)
            for file in os.listdir(record_path):
                file_path = os.path.join(record_path, file)
                self.data.append(file_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]
        # 加载梅尔频率数据
        mel_data = np.load(file_path)  # 假设是.npy格式的文件
        # 对数据进行填充或截断
        if mel_data.shape[0] < self.max_seq_length:
            # 填充到最大长度
            pad_length = self.max_seq_length - mel_data.shape[0]
            mel_data = np.pad(mel_data, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
        elif mel_data.shape[0] > self.max_seq_length:
            # 截断到最大长度
            mel_data = mel_data[:self.max_seq_length, :]
        # 将梅尔频率特征转换为PyTorch张量并返回
        return torch.tensor(mel_data), torch.tensor(label)
