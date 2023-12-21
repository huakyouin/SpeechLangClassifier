import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class MelCNN(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(MelCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 根据传入的input_size动态调整全连接层的输入尺寸
        self.fc_input_features = 64 * (input_size[0] // 4) * (input_size[1] // 4)
        self.fc1 = nn.Linear(self.fc_input_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch,w,h] -> [batch,channel=1,w,h]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # 使用nn.Flatten()层展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class MelTransformer(nn.Module):
    def __init__(self, input_size=80, hidden_size=128, num_layers=10, num_classes=2):
        super(MelTransformer, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=input_size, nhead=4, dim_feedforward=hidden_size, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # 将输入变换为 (seq_len, batch, input_size) 的格式
        output = self.transformer_encoder(x)
        output = output.permute(1, 0, 2)  # 变换回原始的维度顺序
        output = self.fc(output[:, -1, :])  # 取Transformer最后一个时间步的输出作为模型的输出
        return output


