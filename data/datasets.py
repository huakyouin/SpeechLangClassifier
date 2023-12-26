
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

    def get_length_statistics(self):
        lengths = [np.load(file).shape[0] for file in self.data]
        max_length = max(lengths)
        min_length = min(lengths)
        avg_length = sum(lengths) / len(lengths)
        percentile50_length = np.percentile(lengths, 50)
        percentile25_length = np.percentile(lengths, 25)
        percentile75_length = np.percentile(lengths, 75)
        return {
            "max_length": max_length,
            "min_length": min_length,
            "avg_length": avg_length,
            "percentile25_length": percentile25_length,
            "percentile50_length": percentile50_length,
            "percentile75_length": percentile75_length
        }
    
    def cut_left_useless(self,arr):
        # 寻找第一列非空白的位置
        non_negative_index = np.argmax(arr != -11.512925, axis=0)
        # 去除左侧空白列的部分
        trimmed_matrix = arr[ non_negative_index.min():]
        return trimmed_matrix

    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]
        # 加载梅尔频率数据
        mel_data = np.load(file_path)  # 假设是.npy格式的文件
        # 去除左侧空白
        mel_data = self.cut_left_useless(mel_data)
        t_len = mel_data.shape[0]
        # 对数据进行填充或截断
        if mel_data.shape[0] < self.max_seq_length:
            # 填充到最大长度
            pad_length = self.max_seq_length - mel_data.shape[0]
            mel_data = np.pad(mel_data, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
        elif mel_data.shape[0] > self.max_seq_length:
            # 截断到最大长度
            mel_data = mel_data[:self.max_seq_length, :]
        # 将梅尔频率特征转换为PyTorch张量并返回
        return torch.tensor(mel_data), torch.tensor(label) , t_len
