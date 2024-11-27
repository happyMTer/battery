import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import TCNAttentionModel, TCNModel, SimpleRNN, BatteryLSTM
from Data_Preprocessing import data_preprocessing
import csv
import numpy as np

file_path = 'CS2_35'


def train_TCNAttention_Model(file_path):

    X_train, X_test, y_train, y_test = data_preprocessing(file_path)

    # 将数据转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
    # print('将数据转换为PyTorch张量:')
    # print(X_train_tensor.shape)
    # print(y_train_tensor.shape)


    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

    # 实例化模型、损失函数和优化器
    model = TCNAttentionModel(input_dim=X_train.shape[1],
                              hidden_dim=320,
                              output_dim=1,
                              num_channels=[64, 64],
                              kernel_size=2,
                              dropout=0.2)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []

    # 训练模型
    num_epochs = 100
    save_path = 'model_checkpoints/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                # 保存模型的状态字典
                model_path = os.path.join(save_path, f'TCN-Attention_Model_epoch_{epoch + 1}.pth')
                torch.save(model.state_dict(), model_path)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        # 计算当前epoch的平均损失
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)  # 将平均损失添加到列表中

    loss_file_path = 'model_checkpoints/tcn_attention_losses.csv'
    with open(loss_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'Loss'])  # 写入表头
        for epoch, loss in enumerate(losses, start=1):
            csv_writer.writerow([epoch, loss])  # 写入每个epoch的损失

    print(f'TCN-Attention Losses have been saved to {loss_file_path}')


def train_TCN_Model(file_path):

    X_train, X_test, y_train, y_test = data_preprocessing(file_path)

    # 将数据转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
    # print('将数据转换为PyTorch张量:')
    # print(X_train_tensor.shape)
    # print(y_train_tensor.shape)


    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

    # 实例化模型、损失函数和优化器

    model = TCNModel(input_dim=X_train.shape[1],
                     hidden_dim=320,
                     output_dim=1,
                     num_channels=[64, 64],
                     kernel_size=2,
                     dropout=0.2)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []

    # 训练模型
    num_epochs = 100
    save_path = 'model_checkpoints/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for epoch in range(num_epochs):
        running_loss = 0.0  # 用于计算当前epoch的平均损失
        for i, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                # 保存模型的状态字典
                model_path = os.path.join(save_path, f'TCNModel_epoch_{epoch + 1}.pth')
                torch.save(model.state_dict(), model_path)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 计算当前epoch的平均损失
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)  # 将平均损失添加到列表中

    loss_file_path = 'model_checkpoints/tcn_losses.csv'
    with open(loss_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'Loss'])  # 写入表头
        for epoch, loss in enumerate(losses, start=1):
            csv_writer.writerow([epoch, loss])  # 写入每个epoch的损失

    print(f'TCN Losses have been saved to {loss_file_path}')


def train_RNN_Model(file_path):

    X_train, X_test, y_train, y_test = data_preprocessing(file_path)

    # 将数据转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
    # print('将数据转换为PyTorch张量:')
    # print(X_train_tensor.shape)
    # print(y_train_tensor.shape)


    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

    # 实例化模型、损失函数和优化器
    model = SimpleRNN(input_size=5,
                      hidden_size=64,
                      output_size=1,
                      num_layers=1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 初始化空列表保存loss
    losses = []

    # 训练模型
    num_epochs = 100
    save_path = 'model_checkpoints/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for epoch in range(num_epochs):

        running_loss = 0.0  # 用于计算当前epoch的平均损失
        for i, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                # 保存模型的状态字典
                model_path = os.path.join(save_path, f'RNNModel_epoch_{epoch + 1}.pth')
                torch.save(model.state_dict(), model_path)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        # 计算当前epoch的平均损失
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)  # 将平均损失添加到列表中

    loss_file_path = 'model_checkpoints/rnn_losses.csv'
    with open(loss_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'Loss'])  # 写入表头
        for epoch, loss in enumerate(losses, start=1):
            csv_writer.writerow([epoch, loss])  # 写入每个epoch的损失

    print(f'RNN Losses have been saved to {loss_file_path}')



def train_LSTM_Model(file_path):

    X_train, X_test, y_train, y_test = data_preprocessing(file_path)

    # 将数据转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
    # print('将数据转换为PyTorch张量:')
    # print(X_train_tensor.shape)
    # print(y_train_tensor.shape)


    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

    # 实例化模型、损失函数和优化器
    model = BatteryLSTM(input_size=5,
                        hidden_size=64,
                        num_layers=1,
                        output_size=1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 初始化空列表保存loss
    losses = []

    # 训练模型
    num_epochs = 100
    save_path = 'model_checkpoints/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for epoch in range(num_epochs):

        running_loss = 0.0  # 用于计算当前epoch的平均损失
        for i, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                # 保存模型的状态字典
                model_path = os.path.join(save_path, f'LSTMModel_epoch_{epoch + 1}.pth')
                torch.save(model.state_dict(), model_path)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 计算当前epoch的平均损失
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)  # 将平均损失添加到列表中

    loss_file_path = 'model_checkpoints/lstm_losses.csv'
    with open(loss_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'Loss'])  # 写入表头
        for epoch, loss in enumerate(losses, start=1):
            csv_writer.writerow([epoch, loss])  # 写入每个epoch的损失

    print(f'LSTM Losses have been saved to {loss_file_path}')

# train_TCNAttention_Model(file_path)
train_TCN_Model(file_path)
train_RNN_Model(file_path)
train_LSTM_Model(file_path)
