'''
对CALCE数据集进行预处理
'''
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

def data_preprocessing(file_name, shuffle=True):

    # 加载数据
    # Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
    Battery = np.load('./DataSet/CALCE.npy', allow_pickle=True)
    Battery = Battery.item()
    battery = Battery[file_name]

    # 选择特征列 'Cycle_Index', 'Discharge_Capacity(Ah)'等
    features = battery[['cycle', 'capacity', 'SoH', 'resistance', 'CCCT', 'CVCT']]
    # 使用fillna方法来填充NaN值，例如使用0或其他合适的值
    features_filled = features.fillna(0)

    # 设定目标变量:想要预测'Discharge_Capacity(Ah)'
    target = 'capacity'

    # 分离特征和目标变量
    X = features_filled.drop(target, axis=1)
    y = features_filled[target]

    # 数据归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                        y,
                                                        test_size=0.8,
                                                        random_state=42,
                                                        shuffle=shuffle)

    # 因为模型输入需要三维数据，所以需要对数据进行reshape，增加一个时间步长的维度
    # (样本数, 1, 特征数)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    return X_train, X_test, y_train, y_test


def getY(file_name):
    # 加载数据
    # Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
    Battery = np.load('./DataSet/CALCE.npy', allow_pickle=True)
    Battery = Battery.item()
    battery = Battery[file_name]

    # 选择特征列 'Cycle_Index', 'Discharge_Capacity(Ah)'等
    features = battery[['cycle', 'capacity', 'SoH', 'resistance', 'CCCT', 'CVCT']]
    # 使用fillna方法来填充NaN值，例如使用0或其他合适的值
    features_filled = features.fillna(0)

    # 设定目标变量:想要预测'Discharge_Capacity(Ah)'
    target = 'capacity'

    # 分离特征和目标变量
    X = features_filled.drop(target, axis=1)
    y = features_filled[target]

    return y


file_path = './DataSet/CALCE/CS2_35/CS2_35_1_10_11.xlsx'
sheet_name = 'Channel_1-008'
# X_train, X_test, y_train, y_test = data_preprocessing(file_name='CS2_35')
# Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
Battery = np.load('./DataSet/CALCE.npy', allow_pickle=True)
Battery = Battery.item()
print(type(Battery))
# 选择特征列 'Cycle_Index', 'Discharge_Capacity(Ah)'等
battery = Battery['CS2_35']
print(type(battery))
features = battery[['cycle', 'capacity', 'SoH', 'resistance', 'CCCT', 'CVCT']]
print(type(features))


# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
