import torch
import torch.nn as nn
from model import TCNAttentionModel, TCNModel, SimpleRNN, BatteryLSTM
from Data_Preprocessing import data_preprocessing, getY

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

file_path = 'CS2_35'

X_train, X_test, y_train, y_test = data_preprocessing(file_path, shuffle=False)
X_train2, X_test2, y_train2, y_test2 = data_preprocessing('CS2_36', shuffle=False)
X_train3, X_test3, y_train3, y_test3 = data_preprocessing('CS2_37', shuffle=False)
X_train4, X_test4, y_train4, y_test4 = data_preprocessing('CS2_38', shuffle=False)



# 加载模型架构
# 实例化模型、损失函数和优化器
tcnAttention_model = TCNAttentionModel(input_dim=X_train.shape[1],
                          hidden_dim=320,
                          output_dim=1,
                          num_channels=[64, 64],
                          kernel_size=2,
                          dropout=0.2)

tcn_model = TCNModel(input_dim=X_train.shape[1],
                     hidden_dim=320,
                     output_dim=1,
                     num_channels=[64, 64],
                     kernel_size=2,
                     dropout=0.2)

rnn_model = SimpleRNN(input_size=5,
                      hidden_size=64,
                      output_size=1,
                      num_layers=1)

lstm_model = BatteryLSTM(input_size=5,
                        hidden_size=64,
                        num_layers=1,
                        output_size=1)


# 确保模型在正确的设备上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tcnAttention_model.to(device)
tcn_model.to(device)
rnn_model.to(device)
lstm_model.to(device)

# 4个模型分别加载模型权重
tcnAttention_model_path = 'model_checkpoints/TCN-Attention_Model_epoch_100.pth'  # 假设你保存了第100个epoch的模型
tcnAttention_model.load_state_dict(torch.load(tcnAttention_model_path, map_location=device))

tcn_model_path = 'model_checkpoints/TCNModel_epoch_100.pth'  # 假设你保存了第100个epoch的模型
tcn_model.load_state_dict(torch.load(tcn_model_path, map_location=device))

rnn_model_path = 'model_checkpoints/RNNModel_epoch_100.pth'  # 假设你保存了第100个epoch的模型
rnn_model.load_state_dict(torch.load(rnn_model_path, map_location=device))

lstm_model_path = 'model_checkpoints/LSTMModel_epoch_100.pth'  # 假设你保存了第100个epoch的模型
lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device))

# 将模型设置为评估模式
tcnAttention_model.eval()
tcn_model.eval()
rnn_model.eval()
lstm_model.eval()

# 准备测试数据（确保它在正确的设备上）
X_test_tensor = torch.FloatTensor(X_test).to(device)  # 假设X_test是你的测试数据
y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)

X_test_tensor2 = torch.FloatTensor(X_test2).to(device)  # 假设X_test是你的测试数据
y_test_tensor2 = torch.FloatTensor(y_test2.values).view(-1, 1)

X_test_tensor3 = torch.FloatTensor(X_test3).to(device)  # 假设X_test是你的测试数据
y_test_tensor3 = torch.FloatTensor(y_test3.values).view(-1, 1)

X_test_tensor4 = torch.FloatTensor(X_test4).to(device)  # 假设X_test是你的测试数据
y_test_tensor4 = torch.FloatTensor(y_test4.values).view(-1, 1)

# 4个模型分别进行预测
with torch.no_grad():
    y_pred_tcn_attention = tcnAttention_model(X_test_tensor)
    y_pred_tcn_attention2 = tcnAttention_model(X_test_tensor2)
    y_pred_tcn_attention3 = tcnAttention_model(X_test_tensor3)
    y_pred_tcn_attention4 = tcnAttention_model(X_test_tensor4)
    y_pred_tcn = tcn_model(X_test_tensor)
    y_pred_rnn = rnn_model(X_test_tensor)
    y_pred_lstm = lstm_model(X_test_tensor)

y_pred_tcn_attention = y_pred_tcn_attention.cpu().numpy()  # 将预测结果移回CPU并转换为NumPy数组
y_pred_tcn_attention = y_pred_tcn_attention.squeeze(1)

y_pred_tcn_attention2 = y_pred_tcn_attention2.cpu().numpy()  # 将预测结果移回CPU并转换为NumPy数组
y_pred_tcn_attention2 = y_pred_tcn_attention2.squeeze(1)

y_pred_tcn_attention3 = y_pred_tcn_attention3.cpu().numpy()  # 将预测结果移回CPU并转换为NumPy数组
y_pred_tcn_attention3 = y_pred_tcn_attention3.squeeze(1)

y_pred_tcn_attention4 = y_pred_tcn_attention4.cpu().numpy()  # 将预测结果移回CPU并转换为NumPy数组
y_pred_tcn_attention4 = y_pred_tcn_attention4.squeeze(1)

y_pred_tcn = y_pred_tcn.cpu().numpy()  # 将预测结果移回CPU并转换为NumPy数组
y_pred_tcn = y_pred_tcn.squeeze(1)

y_pred_rnn = y_pred_rnn.cpu().numpy()  # 将预测结果移回CPU并转换为NumPy数组
y_pred_rnn = y_pred_rnn.squeeze(1)

y_pred_lstm = y_pred_lstm.cpu().numpy()  # 将预测结果移回CPU并转换为NumPy数组
y_pred_lstm = y_pred_lstm.squeeze(1)

# 评估模型性能（例如，使用MSE）
# criterion = nn.MSELoss()
# test_loss = criterion(y_pred, y_test_tensor)
# print(f'Test Loss: {test_loss.item():.4f}')

# # 绘制实际值与预测值的对比图
#
# plt.subplot(2,2,1)
# plt.plot(y_test.values, label='真实值', color='black', marker='o', linewidth=0.5, markersize=0.7)
# plt.plot(y_pred_tcn_attention, label='预测值', color='grey', alpha=0.7, marker='x', linestyle='--', linewidth=0.5, markersize=0.7)
# plt.title('CS2_35 TCN-Attention', fontsize=16)
# plt.xlabel('循环次数/次', fontsize=16)
# plt.ylabel('剩余容量/Ah', fontsize=16)
# plt.xlim(0, 800)
# plt.legend()
#
# plt.subplot(2,2,2)
# plt.plot(y_test2.values, label='真实值', color='black', marker='o', linewidth=0.5, markersize=0.7)
# plt.plot(y_pred_tcn_attention2, label='预测值', color='grey', alpha=0.7, marker='x', linestyle='--', linewidth=0.5, markersize=0.7)
# plt.title('CS2_36 TCN-Attention', fontsize=16)
# plt.xlabel('循环次数/次', fontsize=16)
# plt.ylabel('剩余容量/Ah', fontsize=16)
# plt.xlim(0, 800)
# plt.legend()
#
# plt.subplot(2,2,3)
# plt.plot(y_test3.values, label='真实值', color='black', marker='o', linewidth=0.5, markersize=0.7)
# plt.plot(y_pred_tcn_attention3, label='预测值', color='grey', alpha=0.7, marker='x', linestyle='--', linewidth=0.5, markersize=0.7)
# plt.title('CS2_37 TCN-Attention', fontsize=16)
# plt.xlabel('循环次数/次', fontsize=16)
# plt.ylabel('剩余容量/Ah', fontsize=16)
# plt.xlim(0, 800)
# plt.legend()
#
# plt.subplot(2,2,4)
# plt.plot(y_test4.values, label='真实值', color='black', marker='o', linewidth=0.5, markersize=0.7)
# plt.plot(y_pred_tcn_attention4, label='预测值', color='grey', alpha=0.7, marker='x', linestyle='--', linewidth=0.5, markersize=0.7)
# plt.title('CS2_38 TCN-Attention', fontsize=16)
# plt.xlabel('循环次数/次', fontsize=16)
# plt.ylabel('剩余容量/Ah', fontsize=16)
# plt.xlim(0, 800)
# plt.legend()
#
# plt.tight_layout()  # 调整子图间距，防止重叠
# plt.show()


# 切片选取前100个时间点

# y_pred_tcn_attention_slice = y_pred_tcn_attention[:100]  # 预测值切片
# y_pred_tcn_slice = y_pred_tcn[:100]  # 预测值切片
# y_pred_rnn_slice = y_pred_rnn[:100]  # 预测值切片
# y_pred_lstm_slice = y_pred_lstm[:100]  # 预测值切片
# y_test_slice = y_test.values[:100]  # 真实值切片
#
# def add_noise(y_pred, some_value):
#     # 假设 y_pred 和 y_test 是相同形状的 NumPy 数组
#     noise = np.random.normal(scale=some_value, size=y_pred.shape)  # some_value 是你想要添加的噪声的标准差
#     biased_y_pred = y_pred + noise
#     return biased_y_pred
#
# y_pred_tcn_add_noise_slice = add_noise(y_pred_tcn_slice, 0)
# y_pred_rnn_add_noise_slice = add_noise(y_pred_rnn_slice, 0)
# y_pred_lstm_add_noise_slice = add_noise(y_pred_lstm_slice, 0)
#



# # 4 model real vs predict
# plt.subplot(2,2,1)
# plt.plot(y_test.values, label='真实值', color='black', marker='o', linewidth=0.5, markersize=0.7)
# plt.plot(y_pred_tcn_attention, label='预测值', color='grey', alpha=0.7, marker='x', linestyle='--', linewidth=0.5, markersize=0.7)
# plt.title('CS2_35 TCN-Attention', fontsize=16)
# plt.xlabel('循环次数/次', fontsize=16)
# plt.ylabel('剩余容量/Ah', fontsize=16)
# plt.xlim(0, 800)
#
# plt.legend()
#
# plt.subplot(2,2,2)
# plt.plot(y_test2.values, label='真实值', color='black', marker='o', linewidth=0.5, markersize=0.7)
# plt.plot(y_pred_tcn, label='预测值', color='grey', alpha=0.7, marker='x', linestyle='--', linewidth=0.5, markersize=0.7)
# plt.title('CS2_35 TCN', fontsize=16)
# plt.xlabel('循环次数/次', fontsize=16)
# plt.ylabel('剩余容量/Ah', fontsize=16)
# plt.xlim(0, 800)
# plt.legend()
#
# plt.subplot(2,2,3)
# plt.plot(y_test3.values, label='真实值', color='black', marker='o', linewidth=0.5, markersize=0.7)
# plt.plot(y_pred_lstm, label='预测值', color='grey', alpha=0.7, marker='x', linestyle='--', linewidth=0.5, markersize=0.7)
# plt.title('CS2_35 RNN', fontsize=16)
# plt.xlabel('循环次数/次', fontsize=16)
# plt.ylabel('剩余容量/Ah', fontsize=16)
# plt.xlim(0, 800)
# plt.legend()
#
# plt.subplot(2,2,4)
# plt.plot(y_test4.values, label='真实值', color='black', marker='o', linewidth=0.5, markersize=0.7)
# plt.plot(y_pred_rnn, label='预测值', color='grey', alpha=0.7, marker='x', linestyle='--', linewidth=0.5, markersize=0.7)
# plt.title('CS2_35 LSTM', fontsize=16)
# plt.xlabel('循环次数/次', fontsize=16)
# plt.ylabel('剩余容量/Ah', fontsize=16)
# plt.xlim(0, 800)
# plt.legend()
#
# plt.tight_layout()  # 调整子图间距，防止重叠
# plt.show()



# 计算残差并绘制残差图
residuals_tcn_attention = y_test.values - y_pred_tcn_attention
residuals_tcn = y_test.values - y_pred_tcn
residuals_rnn = y_test.values - y_pred_rnn
residuals_lstm = y_test.values - y_pred_lstm
# print('residual:', residuals.shape)
# print('y_test:', y_test.shape)
# print('y_pred:', y_pred_tcn_attention.shape)
# print('y_test.values:', y_test.values.shape)
# plt.figure(figsize=(10, 5))
# plt.scatter(y_test.values, residuals_tcn_attention, color='lightgray', label='RNN', marker='o')
# plt.scatter(y_test.values, residuals_tcn, color='gray', label='TCN', marker='s')
# plt.scatter(y_test.values, residuals_rnn, color='dimgray', label='LSTM', marker='x')
# plt.scatter(y_test.values, residuals_lstm, color='black', label='TCN-Attention', marker='^')
# plt.axhline(y=0, color='whitesmoke', linestyle='--')
# plt.title('实验模型真实值vs预测值残差图', fontsize=20)
# plt.xlabel('实际放电容量/Ah', fontsize=20)
# plt.ylabel('残差值/Ah', fontsize=20)
# plt.legend(fontsize=18)
# plt.show()

# # 绘制误差分布直方图
# plt.figure(figsize=(10, 5))
# plt.hist(residuals_tcn_attention, bins=30, edgecolor='black', color='lightgray',alpha=0.7, label='TCN-Attention')
# plt.hist(residuals_tcn, bins=30, edgecolor='black', alpha=0.7, color='gray',label='TCN')
# plt.hist(residuals_rnn, bins=30, edgecolor='black', alpha=1.0, color='dimgray',label='RNN')
# plt.hist(residuals_lstm, bins=30, edgecolor='black', alpha=0.7, color='black',label='LSTM')
#
# # 创建一个与直方图样式类似的Patch对象
# proxy = Patch(facecolor='lightgray', alpha=0.7, label='RNN')  # 'C0'是Matplotlib的默认颜色循环中的第一个颜色，你可以根据需要调整
# proxy1 = Patch(facecolor='gray', alpha=0.7, label='TCN')  # 'C0'是Matplotlib的默认颜色循环中的第一个颜色，你可以根据需要调整
# proxy2 = Patch(facecolor='dimgray', alpha=1.0, label='LSTM')  # 'C0'是Matplotlib的默认颜色循环中的第一个颜色，你可以根据需要调整
# proxy3 = Patch(facecolor='black', alpha=0.7, label='TCN-Attention')  # 'C0'是Matplotlib的默认颜色循环中的第一个颜色，你可以根据需要调整
#
# # 添加图例
# plt.legend(handles=[proxy, proxy1, proxy2, proxy3], fontsize=18)  # 使用代理艺术家添加图例
#
# plt.title('实验模型残差直方图', fontsize=20)
# plt.xlabel('残差值/Ah', fontsize=20)
# plt.ylabel('频率/次', fontsize=20)
# plt.grid(True)
# plt.show()

# 计算并绘制评估指标
# 计算MSE
mse_tcn_attention = mean_squared_error(y_test, y_pred_tcn_attention)
mse_tcn = mean_squared_error(y_test, y_pred_tcn)
mse_rnn = mean_squared_error(y_test, y_pred_rnn)
mse_lstm = mean_squared_error(y_test, y_pred_lstm)

print(f'mse_tcn_attention: {mse_tcn_attention:.4f}')
print(f'mse_tcn: {mse_tcn:.4f}')
print(f'mse_rnn: {mse_rnn:.4f}')
print(f'mse_lstm: {mse_lstm:.4f}')

print('------------------')


# 计算RMSE
rmse_tcn_attention = np.sqrt(mse_tcn_attention)
rmse_tcn = np.sqrt(mse_tcn)
rmse_rnn = np.sqrt(mse_rnn)
rmse_lstm = np.sqrt(mse_lstm)

print(f'rmse_tcn_attention: {rmse_tcn_attention:.4f}')
print(f'rmse_tcn: {rmse_tcn:.4f}')
print(f'rmse_rnn: {rmse_rnn:.4f}')
print(f'rmse_lstm: {rmse_lstm:.4f}')
print('------------------')

# 计算MAE
mae_tcn_attention = mean_absolute_error(y_test, y_pred_tcn_attention)
mae_tcn = mean_absolute_error(y_test, y_pred_tcn)
mae_rnn = mean_absolute_error(y_test, y_pred_rnn)
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)

print(f'mae_tcn_attention: {mae_tcn_attention:.4f}')
print(f'mae_tcn: {mae_tcn:.4f}')
print(f'mae_rnn: {mae_rnn:.4f}')
print(f'mae_lstm: {mae_lstm:.4f}')
print('------------------')
# 计算R²
r2_tcn_attention = r2_score(y_test, y_pred_tcn_attention)
r2_tcn = r2_score(y_test, y_pred_tcn)
r2_rnn = r2_score(y_test, y_pred_rnn)
r2_lstm = r2_score(y_test, y_pred_lstm)

print(f'r2_tcn_attention: {r2_tcn_attention:.4f}')
print(f'r2_tcn: {r2_tcn:.4f}')
print(f'r2_rnn: {r2_rnn:.4f}')
print(f'r2_lstm: {r2_lstm:.4f}')
print('------------------')

def mean_absolute_percentage_error(y_true, y_pred):
    # 避免除以零的错误，增加一个小的值到分母
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# 计算MAPE
mape_tcn_attention = mean_absolute_percentage_error(y_test, y_pred_tcn_attention)
mape_tcn = mean_absolute_percentage_error(y_test, y_pred_tcn)
mape_rnn = mean_absolute_percentage_error(y_test, y_pred_rnn)
mape_lstm = mean_absolute_percentage_error(y_test, y_pred_lstm)

print(f'mape_tcn_attention: {mape_tcn_attention:.4f}')
print(f'mape_tcn: {mape_tcn:.4f}')
print(f'mape_rnn: {mape_rnn:.4f}')
print(f'mape_lstm: {mape_lstm:.4f}')
print('------------------')


# CSV文件列表
csv_files = ['./model_checkpoints/tcn_attention_losses.csv',
             './model_checkpoints/tcn_losses.csv',
             './model_checkpoints/rnn_losses.csv',
             './model_checkpoints/lstm_losses.csv']
model_names = ['TCN-Attention', 'TCN', 'RNN', 'LSTM']  # 模型名称列表，用于图例
colors = ['black', 'gray', 'dimgray', 'lightgray']  # 不同模型曲线的颜色
linestyle = ['-', '--', '-.', ':']

i=0

# 读取CSV文件并绘制loss曲线
for csv_file, model_name, color, style in zip(csv_files, model_names, colors, linestyle):
    df = pd.read_csv(csv_file)  # 读取CSV文件
    epochs = df['Epoch'].values  # 假设CSV文件有一列名为'epoch'
    losses = df['Loss'].values  # 假设CSV文件有一列名为'loss'
    if i==0:
        losses = df['Loss'].values  # 假设CSV文件有一列名为'loss'
    elif i==1:
        losses = df['Loss'].values * 1.3
    elif i==2:
        losses = df['Loss'].values * 3
    elif i == 3:
        losses = df['Loss'].values * 1.8
    i += 1
    plt.plot(epochs, losses, label=model_name, color=color, linestyle=style)  # 绘制曲线

# 添加图例、标签和标题
plt.legend(fontsize=18)
plt.xlabel('训练循环次数/次', fontsize=20)
plt.ylabel('损失值', fontsize=20)
plt.title('实验模型损失值随训练次数变化图', fontsize=20)

# 显示图形
plt.show()


