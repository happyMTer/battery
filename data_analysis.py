import glob

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.cm as cm


def curr_vol_cycle_change():

    # 加载CS2_35数据
    df = pd.read_excel('./DataSet/CALCE/CS2_35/CS2_35_1_10_11.xlsx', sheet_name='Channel_1-008')
    # 选择特征列 'Cycle_Index', 'Discharge_Capacity(Ah)'等
    features_35 = df[['Data_Point', 'Cycle_Index', 'Current(A)', 'Voltage(V)', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)',
                   'Internal_Resistance(Ohm)']].copy()

    current_35_at_cycle_1 = features_35.loc[features_35['Cycle_Index'] == 1, 'Current(A)']
    voltage_35_at_cycle_1 = features_35.loc[features_35['Cycle_Index'] == 1, 'Voltage(V)']
    index_35_at_cycle_1 = features_35.loc[features_35['Cycle_Index'] == 1, 'Data_Point']



    # 加载CS2_36数据
    df = pd.read_excel('./DataSet/CALCE/CS2_36/CS2_36_1_10_11.xlsx', sheet_name='Channel_1-009')
    # 选择特征列 'Cycle_Index', 'Discharge_Capacity(Ah)'等
    features_36 = df[['Data_Point', 'Cycle_Index', 'Current(A)', 'Voltage(V)', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)',
                   'Internal_Resistance(Ohm)']].copy()

    current_36_at_cycle_1 = features_36.loc[features_36['Cycle_Index'] == 1, 'Current(A)']
    voltage_36_at_cycle_1 = features_36.loc[features_36['Cycle_Index'] == 1, 'Voltage(V)']
    index_36_at_cycle_1 = features_36.loc[features_36['Cycle_Index'] == 1, 'Data_Point']

    # 加载CS2_37数据
    df = pd.read_excel('./DataSet/CALCE/CS2_37/CS2_37_1_10_11.xlsx', sheet_name='Channel_1-010')
    # 选择特征列 'Cycle_Index', 'Discharge_Capacity(Ah)'等
    features_37 = df[['Data_Point', 'Cycle_Index', 'Current(A)', 'Voltage(V)', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)',
                   'Internal_Resistance(Ohm)']].copy()

    current_37_at_cycle_1 = features_37.loc[features_37['Cycle_Index'] == 1, 'Current(A)']
    voltage_37_at_cycle_1 = features_37.loc[features_37['Cycle_Index'] == 1, 'Voltage(V)']
    index_37_at_cycle_1 = features_37.loc[features_37['Cycle_Index'] == 1, 'Data_Point']

    # 加载CS2_38数据
    df = pd.read_excel('./DataSet/CALCE/CS2_38/CS2_38_1_10_11.xlsx', sheet_name='Channel_1-011')
    # 选择特征列 'Cycle_Index', 'Discharge_Capacity(Ah)'等
    features_38 = df[['Data_Point', 'Cycle_Index', 'Current(A)', 'Voltage(V)', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)',
                   'Internal_Resistance(Ohm)']].copy()

    current_38_at_cycle_1 = features_38.loc[features_38['Cycle_Index'] == 1, 'Current(A)']
    voltage_38_at_cycle_1 = features_38.loc[features_38['Cycle_Index'] == 1, 'Voltage(V)']
    index_38_at_cycle_1 = features_38.loc[features_38['Cycle_Index'] == 1, 'Data_Point']



    # 创建一个2x2的子图网格
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # 定义一个函数来简化双轴图的创建
    def plot_with_twin_axes(ax, index, voltage_data, current_data, title):
        ax.plot(index, voltage_data, 'r', label='Voltage')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)', color='r')
        ax.tick_params('y', colors='r')
        ax.legend(loc='upper left')

        ax_twin = ax.twinx()
        ax_twin.plot(index, current_data, 'b', label='Current')
        ax_twin.set_ylabel('Current (A)', color='b')
        ax_twin.tick_params('y', colors='b')
        ax_twin.legend(loc='upper right')
        ax.set_title(title)


    # 使用定义的函数来绘制每个子图
    plot_with_twin_axes(axs[0, 0],
                        index_35_at_cycle_1.values,
                        voltage_35_at_cycle_1.values,
                        current_35_at_cycle_1.values,
                        'CS2_35 Cycle 1')
    plot_with_twin_axes(axs[0, 1],
                        index_36_at_cycle_1.values,
                        voltage_36_at_cycle_1.values,
                        current_36_at_cycle_1.values,
                        'CS2_36 Cycle 1')
    plot_with_twin_axes(axs[1, 0],
                        index_37_at_cycle_1.values,
                        voltage_37_at_cycle_1.values,
                        current_37_at_cycle_1.values,
                        'CS2_37 Cycle 1')
    plot_with_twin_axes(axs[1, 1],
                        index_38_at_cycle_1.values,
                        voltage_38_at_cycle_1.values,
                        current_38_at_cycle_1.values,
                        'CS2_38 Cycle 1')

    # 调整子图之间的距离
    plt.tight_layout()
    # 显示图形
    plt.show()


def SOH_Analysis():
    # 加载数据
    Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
    Battery = np.load('./DataSet/CALCE.npy', allow_pickle=True)
    Battery = Battery.item()

    # 放电容量 & 放电周期
    # Rated_Capacity = 1.1
    fig, ax = plt.subplots(1, figsize=(12, 8))
    # color_list = ['b:', 'g--', 'r-.', 'c:']
    for name in zip(Battery_list):
        print(name[0])
        battery = Battery[name[0]]

        ax.plot(battery['cycle'].values, battery['capacity'].values, label='Battery_' + name[0])
    # plt.plot([-1,1000],[Rated_Capacity*0.7, Rated_Capacity*0.7], c='black', lw=1, ls='--')  # 临界点直线
    ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)',
           title='Capacity degradation at ambient temperature of 1°C')
    plt.legend()
    plt.show()

    # 放电容量&内阻变化
    battery1 = Battery['CS2_35']
    battery2 = Battery['CS2_36']
    battery3 = Battery['CS2_37']
    battery4 = Battery['CS2_38']

    # 创建一个2x2的子图网格，并获取figure和axes对象
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # 绘制第一个子图
    axs[0, 0].scatter(battery1['cycle'], battery1['SoH'], c=battery1['resistance'], s=10, cmap=cm.plasma)
    axs[0, 0].set_xlabel('Number of Cycles', fontsize=14)
    axs[0, 0].set_ylabel('State of Health', fontsize=14)
    cbar1 = fig.colorbar(axs[0, 0].collections[0], ax=axs[0, 0])
    cbar1.set_label('Internal Resistance (Ohm)', fontsize=14, rotation=-90, labelpad=20)
    axs[0, 0].set_title('CS2_35')

    # 绘制第二个子图
    axs[0, 1].scatter(battery2['cycle'], battery2['SoH'], c=battery2['resistance'], s=10, cmap=cm.plasma)
    axs[0, 1].set_xlabel('Number of Cycles', fontsize=14)
    axs[0, 1].set_ylabel('State of Health', fontsize=14)
    cbar2 = fig.colorbar(axs[0, 1].collections[0], ax=axs[0, 1])
    cbar2.set_label('Internal Resistance (Ohm)', fontsize=14, rotation=-90, labelpad=20)
    axs[0, 1].set_title('CS2_36')

    # 绘制第三个子图
    axs[1, 0].scatter(battery3['cycle'], battery3['SoH'], c=battery3['resistance'], s=10, cmap=cm.plasma)
    axs[1, 0].set_xlabel('Number of Cycles', fontsize=14)
    axs[1, 0].set_ylabel('State of Health', fontsize=14)
    cbar3 = fig.colorbar(axs[1, 0].collections[0], ax=axs[1, 0])
    cbar3.set_label('Internal Resistance (Ohm)', fontsize=14, rotation=-90, labelpad=20)
    axs[1, 0].set_title('CS2_37')

    # 绘制第四个子图
    axs[1, 1].scatter(battery4['cycle'], battery4['SoH'], c=battery4['resistance'], s=10, cmap=cm.plasma)
    axs[1, 1].set_xlabel('Number of Cycles', fontsize=14)
    axs[1, 1].set_ylabel('State of Health', fontsize=14)
    cbar4 = fig.colorbar(axs[1, 1].collections[0], ax=axs[1, 1])
    cbar4.set_label('Internal Resistance (Ohm)', fontsize=14, rotation=-90, labelpad=20)
    axs[1, 1].set_title('CS2_38')

    # 调整子图之间的距离
    plt.tight_layout()

    # 显示图形
    plt.show()

    # plt.figure(figsize=(9, 6))
    # plt.scatter(battery['cycle'], battery['SoH'], c=battery['resistance'], s=10)
    # cbar = plt.colorbar()
    # cbar.set_label('Internal Resistance (Ohm)', fontsize=14, rotation=-90, labelpad=20)
    # plt.xlabel('Number of Cycles', fontsize=14)
    # plt.ylabel('State of Health', fontsize=14)
    # plt.legend()
    # plt.show()

    # 各项指标 & 充放电周期
    battery = Battery['CS2_35']
    plt.figure(figsize=(12, 9))
    names = ['capacity', 'resistance', 'CCCT', 'CVCT']
    colors = ['red', 'green', 'blue', 'purple']  # 定义四种颜色用于四个子图
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.scatter(battery['cycle'], battery[names[i]], s=10, color=colors[i])
        plt.xlabel('Number of Cycles', fontsize=14)
        plt.ylabel(names[i], fontsize=14)
    plt.legend()
    plt.show()
SOH_Analysis()



