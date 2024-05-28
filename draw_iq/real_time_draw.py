"""ToDo
为实时系统设计的画图程序  √
输入的信号数据为float32的比特流数据  √
x的尺度保持相对不变  √
固定y轴的尺度  ×
"""
import shutil
import math
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, windows
import os
targetFolderPath = 'C:/signal_sample_X310/2024.5.09/'  # 设置数据的基础路径
figure_OutPath = 'C:/Users/user/Desktop/test/'  # 设置图片的保存路径
file_name = 'data.bin_75000000SPS_-2147483648Hz_2024_04_09_T16-16-46.iq'  # 给单次画图用的


drone_name = 'test'
mode = 2  # 模式1是连续画图，模式2是单次画图
time_duration = 0.5  # 持续时间(x轴)
save_time = 2   # usrp采集时间
fs = 100e6  # 采样率，15e6是B210的采样率，75e6是X310的采样率，100e6是原数据集的采样率，固定尺度要注意固定在100MHZ这个尺度上
slice_point = int(fs * time_duration)
stft_point = 1024  # stft点数和采样率共同决定了y轴的分辨率，但在一般情况下整个系统在y轴上的分辨率相对固定的，只需要变x轴的分辨率
slice_point_rate = int(fs * save_time)
y_range = (2400, 2500)  # y轴标尺的变化范围


# 多次画图
def mode1():
    while True:
        files = os.listdir(targetFolderPath)
        for file in files:
            with open(targetFolderPath + file) as fp:
                file_size = os.path.getsize(targetFolderPath + file)
                read_time = int(np.ceil(file_size / 2 / slice_point)) - 2
                read_data = np.fromfile(fp, dtype=np.int16)
                for i in range(read_time):
                    dataI = read_data[::2]
                    if len(dataI) >= slice_point:
                        # 绘制 dataI 的 stft 图像
                        # stft点数在频域上类似与画图用的采样率，虽然用了那么一包数据去画一张图，但也需要stft点数足够才能还原出真实的时频图情况，
                        # 这里的概念就和采样信号一样
                        f, t, Zxx = stft(dataI[i * slice_point:(i + 1) * slice_point],
                                            fs, window=windows.hamming(stft_point), nperseg=stft_point)
                        plt.figure()
                        plt.pcolormesh(t, f, 20 * np.log10(np.abs(Zxx)), cmap='jet')  # 获取复数的绝对值
                        plt.title("Mavic2 " + str(i))
                        plt.savefig(figure_OutPath + drone_name + str(i) + '.jpg', dpi=300)
                        plt.close()  # 转opencv格式要把这个放到后面
                        """
                        fig = plt.gcf()  # 获取当前图形
                        fig.canvas.draw()  # 绘制图形到画布

                            # 获取图像数组并转换为OpenCV格式
                        img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # 转换为RGB图像数据
                        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # 重塑为图像形状
                        """


# 单次画图
def mode2():
    with open(targetFolderPath + file_name) as fp:
        file_size = os.path.getsize(targetFolderPath + file_name)
        read_time = int(np.ceil(file_size / 2 / slice_point))
        read_data = np.fromfile(fp, dtype=np.float32)
        dataI = read_data[::2]
        save_rate = len(dataI)*2 / slice_point_rate
        for i in range(read_time):
            if len(dataI) >= slice_point:


                # 绘制 dataI 的 stft 图像
                start_time_STFT = time.time()
                f, t, Zxx = stft(dataI[i * slice_point:(i + 1) * slice_point],
                                 fs, window=windows.hamming(stft_point), nperseg=stft_point)
                end_time_STFT = time.time()
                execution_time_STFT = end_time_STFT - start_time_STFT

                start_time_plot = time.time()
                plt.figure(figsize=(19 * time_duration * 10, 10))
                x = 20 * np.log10(np.abs(Zxx))
                plt.pcolormesh(t, f, x, cmap='jet')  # 获取复数的绝对值
                # plt.ylim(2440e6, 2460e6)  # max(amplitudes) 是幅度数据的最大值
                # 设置 y 轴刻度值
                plt.yticks([0, 1e7, 2e7, 3e7, 4e7, 5e7], ['2440 MHz', '2445 MHz', '2450 MHz', '2455 MHz', '2460 MHz', '2465 MHz'])
                # plt.title(drone_name + str(i))
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                end_time_plot = time.time()
                execution_time_plot = end_time_plot - start_time_plot


                plt.savefig(figure_OutPath + drone_name + str(i) + '.jpg', bbox_inches='tight')
                plt.show()
                plt.close()

                print(save_rate, execution_time_STFT, execution_time_plot)


def main():
    if mode == 1:
        mode1()
    elif mode == 2:
        mode2()


if __name__ == '__main__':
    main()