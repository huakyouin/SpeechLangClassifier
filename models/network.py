import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class MelRNN(nn.Module):
    def __init__(self, input_size=80, hidden_size=128, num_layers=2, num_classes=2):
        super(MelRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)  # 初始化隐藏状态
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # 取RNN最后一个时间步的输出作为模型的输出
        return out
    
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


