'''
TCN-Attention model
'''
# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义一个Chomp1d类，用于删除卷积后产生的多余的padding
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()  # 调用父类的初始化方法
        self.chomp_size = chomp_size  # 要删除的数据量

    def forward(self, x):
        # 删除最后一个维度上的chomp_size个数据，并确保数据是连续的
        return x[:, :, :-self.chomp_size].contiguous()


# 定义一个TemporalBlock类，它是TCN的基本构建块
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        '''
        :param n_inputs: 输入通道数（特征数）
        :param n_outputs: 输出通道数（特征数），即卷积中的过滤器数量，每个过滤器都会学习并提取输入数据中的某种特定特征
        :param kernel_size: 卷积核的大小，即过滤器的大小
        :param stride:  卷积核在输入数据上移动的步长
        :param dilation:    卷积核中的元素之间的距离
                            例如，dilation=1 表示卷积核中的元素是连续的，
                            而 dilation=2 表示卷积核中的每个元素之间都有一个空格
                            这可以用于增加卷积核的感受野，而不增加参数数量或计算复杂度
        :param padding: 在输入数据的边缘添加的零填充的数量。这有助于控制输出数据的大小
                        例如，当 kernel_size=3 时，如果 padding=1，则在输入数据的开始和结束处都会添加一个零，
                        这样可以确保输出数据和输入数据具有相同的大小（在 stride=1 的情况下）。
        :param dropout:
        '''
        super(TemporalBlock, self).__init__()  # 调用父类的初始化方法
        # 定义两个一维卷积层，中间包含ReLU激活函数和Dropout
        self.conv1 = nn.Conv1d(n_inputs,
                               n_outputs,
                               kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation)
        self.chomp1 = Chomp1d(padding)  # 删除卷积后多余的padding
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.dropout1 = nn.Dropout(dropout)  # Dropout层，防止过拟合

        self.conv2 = nn.Conv1d(n_outputs,
                               n_outputs,
                               kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 将以上层顺序组合成一个序列模型
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 如果输入和输出的通道数不同，则添加一个1x1的卷积来进行通道数的变换
        self.downsample = nn.Conv1d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()  # 最后的ReLU激活函数
        self.init_weights()  # 初始化权重

    def init_weights(self):
        # 使用正态分布初始化卷积层的权重
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):

        out = self.net(x)  # 数据通过卷积网络
        # print('数据通过TCN net后的形状:',out.shape)
        # 如果输入输出通道数不同，进行下采样
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)  # 残差连接和ReLU激活


# 定义一个TCN类，它由多个TemporalBlock组成
class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        '''
        :param num_inputs: 输入数据通道数（特征数）
        :param num_channels: 每层的输出通道数（特征数）
        :param kernel_size: 卷积核大小
        :param dropout:
        '''
        super(TCN, self).__init__()  # 调用父类的初始化方法
        layers = []  # 用于存储TemporalBlock的列表
        num_levels = len(num_channels)  # TCN的层数
        for i in range(num_levels):
            dilation_size = 2 ** i  # 每一层的膨胀系数
            # 根据是否是第一层来确定输入和输出通道数
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            # 添加TemporalBlock到layers列表中
            layers.append(TemporalBlock(in_channels,
                                        out_channels,
                                        kernel_size,
                                        stride=1,
                                        dilation=dilation_size,
                                        padding=(kernel_size - 1) * dilation_size,
                                        dropout=dropout))
            # 将layers列表中的层顺序组合成一个序列模型
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # print('network之前x.shape=',x.shape)
        return self.network(x)  # 数据通过TCN网络


# # 定义一个Attention类，用于实现注意力机制
# class Attention(nn.Module):
#     def __init__(self, feature_dim, **kwargs):
#         super(Attention, self).__init__(**kwargs)
#         self.feature_dim = feature_dim
#         # 初始化注意力权重，权重维度为(feature_dim,)
#         self.weights = nn.Parameter(torch.rand(feature_dim))
#
#     def forward(self, x):
#         # 假设x的维度是(batch_size, seq_len, feature_dim)
#         batch_size, seq_len, _ = x.size()
#         # 将weights扩展到(batch_size, 1, feature_dim)，以便与x进行广播乘法
#         weights = self.weights.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, self.feature_dim)
#         print('Attention中 weights.size=',weights.shape)
#         print('Attention中 x.size=',x.shape)
#         # 计算注意力分数
#         eij = torch.bmm(x, weights.transpose(1, 2)).squeeze(2)  # batch_size * seq_len
#         # 使用softmax函数对注意力分数进行归一化
#         alpha = F.softmax(eij, dim=1).unsqueeze(2)  # softmax over seq_len, add dim for broadcast
#         # 使用注意力权重对输入进行加权
#         attention_output = x * alpha  # element-wise multiplication, broadcasting occurs here
#         # 对加权后的输入进行求和，得到最终的注意力表示
#         return torch.sum(attention_output, dim=1)  # sum over seq_len





# 定义一个标准点积注意力机制
class DotProductAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value, mask=None):
        # query, key, value的形状都是[batch_size, seq_len, features]
        batch_size = query.size(0)
        # 计算query和key的点积
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(value.size(-1), dtype=query.dtype))

        if mask is not None:
            # Apply the mask to the scores
            scores = scores.masked_fill(mask == 0, -1e9)

            # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to the values
        context = torch.matmul(attention_weights, value)

        return context, attention_weights



# 定义一个TCNAttentionModel类，将TCN和Attention组合在一起
class TCNAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_channels, kernel_size, dropout):
        super(TCNAttentionModel, self).__init__()  # 调用父类的初始化方法
        self.tcn = TCN(input_dim, num_channels, kernel_size=kernel_size, dropout=dropout)  # TCN网络
        # Attention层，hidden_dim应与TCN的最后一个通道数相匹配
        self.attention = DotProductAttention(dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)  # 全连接层，用于输出预测结果

    def forward(self, x):
        x = self.tcn(x)  # 数据先通过TCN网络

        # x的形状是[batch_size, channels, features]
        # 直接将其作为Query, Key, Value传递给Attention层
        x, attention_weights = self.attention(x, x, x)
        # print('经过Attention后的x的形状：',x.shape)
        # print('Attention中weights的形状：',attention_weights.shape)

        # 展平TCN的输出以便传递给全连接层
        x = x.view(x.size(0), -1)  # 将(batch_size, seq_len, channels)展平为(batch_size, -1)
        # print('展平后的形状:', x.shape)

        x = self.fc(x)  # 最后通过全连接层进行预测
        return x  # 返回预测结果


# 定义一个纯TCN模型
class TCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        self.tcn = TCN(input_dim, num_channels, kernel_size=kernel_size, dropout=dropout)  # TCN网络
        self.fc = nn.Linear(hidden_dim, output_dim)  # 全连接层，用于输出预测结果


    def forward(self, x):
        x = self.tcn(x)  # 数据先通过TCN网络
        # 展平TCN的输出以便传递给全连接层
        x = x.view(x.size(0), -1)  # 将(batch_size, seq_len, channels)展平为(batch_size, -1)
        # print('展平后的形状:', x.shape)
        x = self.fc(x)  # 最后通过全连接层进行预测
        return x  # 返回预测结果


# 定义一个RNN模型
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播RNN
        out, _ = self.rnn(x, h0)

        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


# 定义一个LSTM模型
class BatteryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BatteryLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和单元状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out
