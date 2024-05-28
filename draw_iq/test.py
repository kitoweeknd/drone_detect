
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import stft, windows


time_duration = 0.1  # 预设时间窗的长度(关键参数)，决定x尺度
fs = 100e6  # 采样带宽要和采样设备匹配,实验室设备的采样带宽一般为15MHZ，原论文中的最大采样带宽为100MHZ，采样带宽决定y轴尺度
slice_point = int(fs * time_duration)
fly_name = 'Fubuta'  # 预设无人机的名称
distance = '111'
coloar_bar = False
stft_point = 2048  # 采样点数是每次y轴用的点数,1024就够用了
targetFolderPath = 'C:/Users/user/Desktop/11/'


def main():
    i = 0
    for i in range(10):
        filePath = 'E:/Drone_dataset/RFA/DroneRFa/FutabaT14SG_FLY/low/T10110_S0101.mat'
        data = h5py.File(filePath, 'r')
        data_I1 = data['RF0_I'][0]
        # data_Q1 = data['RF0_Q'][0]  # 同一组信号的IQ读一路就行
        data_I2 = data['RF1_I'][0]
        # data_Q2 = data['RF1_I'][0]  # 同一组信号的IQ读一路就行
        j = 0
        while (j+3) * slice_point <= len(data_I1):
            # 先画第一个通道的
            f_I1, t_I1, Zxx_I1 = stft(data_I1[i*slice_point: (i+1) * slice_point],
                             fs, window=windows.hamming(stft_point), nperseg=stft_point)
            augmentation_Zxx1 = 20*np.log10(np.abs(Zxx_I1))  # 是否选择增强

            # 画第一组IQ
            # plt.figure()
            plt.ioff()  # 关闭可视窗
            plt.pcolormesh(t_I1, f_I1, augmentation_Zxx1, cmap='jet')  # 画图时在颜色标尺上应用了一个软件增益使信号在图中更加明显，其实不用也行，用了还会丢失距离信息
            if coloar_bar:
                plt.colorbar()  # 在做实验观察信号的时候可以加一个颜色标尺，做数据集时不用加颜色标尺
            plt.title(fly_name + (str(i)))
            plt.savefig(targetFolderPath + fly_name + str(i) + distance + '985MHZ.jpg', dpi=300)  # dpi设置了图片保存的清晰度，300对应1K的图片
            plt.close()

            # 再画第二个通道的
            f_I2, t_I2, Zxx_I2 = stft(data_I2[i*slice_point: (i+1) * slice_point],
                             fs, window=windows.hamming(stft_point), nperseg=stft_point)
            augmentation_Zxx2 = 20*np.log10(np.abs(Zxx_I2))  # 是否选择增强

            # 画第二组IQ
            plt.figure()
            plt.pcolormesh(t_I2, f_I2, augmentation_Zxx2, cmap='jet')  # 画图时在颜色标尺上应用了一个软件增益使信号在图中更加明显，其实不用也行，用了还会丢失距离信息
            if coloar_bar:
                plt.colorbar()  # 在做实验观察信号的时候可以加一个颜色标尺，做数据集时不用加颜色标尺
            plt.title(fly_name + (str(i)))
            plt.savefig(targetFolderPath + fly_name + str(i) + distance + '2.4GHZ.jpg', dpi=300)
            plt.close()
            j += 1
    return 0


if __name__ == '__main__':
    main()