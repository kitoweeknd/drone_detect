import shutil
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, windows
import os


# 在频率上写一个截取算法，统一尺度，当采样率不够的时候要有一个截取算法
# 要把所有文件用.dat来读，速度很快，plt转opencv的方式再试一试非常影响运算速度
targetFolderPath = 'C:/Users/user/Desktop/11/'  # 设置数据的基础路径
figure_OutPath = 'C:/Users/user/Desktop/11/'  # 设置图片的保存路径
file_name = 'low.dat'  # 给单次画图用的


drone_name = 'empty'
mode = 2  # 模式1是连续画图模式，模式2是单次画图模式
time_duration = 0.1  # 0.03
fs = 100e6  # 15e6是B210的采样率，75e6是X310的采样率，100e6是原数据集的采样率
slice_point = int(fs * time_duration)
stft_point = 2048


# 多次画图
def mode1():
    while True:
        files = os.listdir(targetFolderPath)
        for file in files:
            with open(targetFolderPath + file) as fp:
                file_size = os.path.getsize(targetFolderPath + file)
                read_time = int(np.ceil(file_size / 2 / slice_point)) - 2
                read_data = np.fromfile(fp, dtype=np.int32)
                for i in range(read_time):
                    dataI = read_data[::2]
                    if len(dataI) >= slice_point:
                        # 绘制 dataI 的 stft 图像
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
        read_time = int(np.ceil(file_size / 2 / slice_point)) - 1
        read_data = np.fromfile(fp, dtype=np.float64)
        for i in range(read_time):
            dataI = read_data[::2]
            if len(dataI) >= slice_point:
                # 绘制 dataI 的 stft 图像
                f, t, Zxx = stft(dataI[i * slice_point:(i + 1) * slice_point],
                                 fs, window=windows.hamming(stft_point), nperseg=stft_point)
                plt.figure()
                plt.pcolormesh(t, f, 20 * np.log10(np.abs(Zxx)), cmap='jet')  # 获取复数的绝对值
                plt.title(drone_name + str(i))
                plt.savefig(figure_OutPath + drone_name + str(i) + '.jpg', dpi=300)
                plt.close()


def main():
    if mode == 1:
        mode1()
    elif mode == 2:
        mode2()


if __name__ == '__main__':
    main()