import matplotlib.pyplot as plt
from scipy.signal import stft, windows
import numpy as np

fs = 30e6   # 实验室里的USRP最适合24MHZ加0.1s，雷达采的采样率为30MHZ
stft_point = 2048
duration_time = 0.1
slice_point = int(fs * duration_time)
fig_save_path = 'E:/Drone_dataset/雷达设备专采/转换后/实验/'
file_path = 'E:/Drone_dataset/雷达设备专采/转换后/实验/20240322164025.Bin'


with open(file_path, 'rb') as fp:
    read_data = np.fromfile(fp, dtype=np.int16)  # 这里解析文件要统一为int16
    dataI = read_data[::2]
    if len(dataI) >= slice_point:
        # 绘制 dataI 的 stft 图像
        for ii in range(len(dataI) // slice_point):
            f, t, Zxx = stft(dataI[ii * slice_point:(ii + 1) * slice_point],
                             fs, window=windows.hamming(stft_point), nperseg=stft_point)

            plt.figure()
            plt.pcolormesh(t, f, 20 * np.log10(np.abs(Zxx)))  # 获取复数的绝对值
            plt.title("I " + str(ii))
            # 显示图像
            plt.savefig(fig_save_path + 'temp')
            plt.close()

            """"
            #这里的转格式算法要重新找一个
            fig = plt.gcf()  # 获取当前图形
            fig.canvas.draw()  # 绘制图形到画布
            # 获取图像数组并转换为OpenCV格式
            img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # 转换为RGB图像数据
            img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # 重塑为图像形状
            # 转换为OpenCV的BGR格式
            img_data_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)  # RGB转BGR
            """