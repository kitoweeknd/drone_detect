"""
改模型路径
改share路径
改res_path
改保存图片的路径
所有路径都不能有中文
输入的信号是MHZ


TODO
1.把时间标尺改成一个连续变化的 ❌
2.确定一下持续时间和采样率，对于全频谱的检测可以考虑对全频谱切片切成6片按顺序送到模型里判断 ⭕ 持续时间和采样率要可以还原出原信号
3.确定一下颜色标尺用一个固定变化率的颜色标尺
4.确定一下用不用软件增益 ⭕ 用
5.画图时的命名方式确定一下 ⭕
6.重采样 ❌不需要由于信号的特征能否映射到图片上取决于采集设备的采样带宽是否够

论文研究内容:
7.信噪比作为颜色标尺
8.AWG信噪比加噪声做对比实验

写一个截取算法，统一尺度，当采样率不够的时候要有一个截取算法，在频率上
要把所有文件用.dat来读，速度很快


采样率*时间长度为每次从IQ数据中计算拿的点数
采样率和时间长度一定要固定，如果采样率改变，时间长度也要变，要使每次拿的采样点数都差不多
要改图片的尺度的话只能改stft点数
但信号在时间和频率上的变化特征很大程度上取决于采样设备的采样带宽，画图时保持这些尺度不变，能还原出信号即可
"""


import matplotlib.pyplot as plt
from scipy.signal import stft, windows
import sys
import argparse
from pathlib import Path
import os
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import time
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.general import (cv2, non_max_suppression)


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # 当前路径的上一级目录
ROOT_dat = str(FILE.parents[0])
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = str(ROOT)


stft_point = 2048
time_duration = 0.1


def create_folders(target_path):
    # 创建link文件夹
    link_path = os.path.join(target_path, 'link')
    os.makedirs(link_path, exist_ok=True)

    # 在link文件夹下创建三个子文件夹
    subfolders = ['share', 'res', 'save_fig']
    for folder in subfolders:
        folder_path = os.path.join(link_path, folder)
        os.makedirs(folder_path, exist_ok=True)


def main():
    create_folders(ROOT)
    temp = ROOT.replace('\\', '/')
    directory = temp + '/link/' + 'share/'
    res_path = temp + '/link/' + 'res/'
    fig_save_path = temp + '/link/' + 'save_fig/'
    while True:
        # 检测是否存在txt文件
        files = [f for f in os.listdir(directory) if f.endswith('.txt')]
        if files:
            # 读取第一个txt文件的内容并执行算法
            temp = os.path.join(directory, files[0])
            try:
                with open(temp, 'r', encoding='utf-8') as file:
                    file_path = file.readline().strip()
                    fs = int(file.readline().strip()) * 10e5
            except UnicodeDecodeError:
                # 如果 'utf-8' 编码失败，尝试其他编码
                with open(temp, 'r', encoding='gbk') as file:
                    file_path = file.readline().strip()
                    fs = int(file.readline().strip()) * 10e6

            iq_draw(file_path, res_path, fig_save_path, fs)
            os.remove(temp)
            # print("算法执行完毕，并将结果写入 result.txt 文件。")
        else:
            # print("未检测到txt文件，继续等待...")
            time.sleep(1)  # 每秒检测一次


def iq_draw(file_path, res_path, fig_save_path, fs):
    with open(file_path, 'rb') as fp:
        read_data = np.fromfile(fp, dtype=np.int16)
        dataI = read_data[::2]
        slice_point = int(fs * time_duration)
        if len(dataI) >= slice_point:
            # 绘制 dataI 的 stft 图像
            for ii in range(len(dataI) // slice_point):
                f, t, Zxx = stft(dataI[ii * slice_point:(ii + 1) * slice_point],
                                    fs, window=windows.hamming(stft_point), nperseg=stft_point)

                plt.figure()
                plt.pcolormesh(t, f, 20*np.log10(np.abs(Zxx)))  # 获取复数的绝对值
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
                return inference(file_path, res_path, fig_save_path)



def inference(file_path, res_path, fig_save_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default=ROOT + '\\' + 'drone.pt',
                        help='model.pt path(s)')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    device = torch.device('cpu')
    # device = torch.device('cuda:0')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    cudnn.benchmark = True
    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)  # load FP32 model
    if half:
        model.half()  # to FP16
    # Get names and colors
    names = model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    name_list = []  # 一次预测的结果
    with torch.no_grad():
        showimg = cv2.imread(fig_save_path + 'temp.png')
        # cv2.imshow('test', showimg)
        # cv2.waitKey(0)
        img = showimg
        img = letterbox(img, new_shape=opt.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                   agnostic=opt.agnostic_nms)
        print(pred)
        # Process detections
        for i, det in enumerate(pred):
            res = {}
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                res[names[int(c)]] = int(n)
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], showimg.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (names[int(cls)], conf)
                    name_list.append(names[int(cls)])
                    print(label)
                    plot_one_box(
                        xyxy, showimg, label=label, color=colors[int(cls)], line_thickness=2)
        # Process detections
        # Process detections
        if 'MavicAir2S' in res:
            print('检测到MavicAir2S')
            s = "find MavicAir2S!"  # 替换为您的语音文本
        elif 'AVATA' in res:
            print('检测到AVATA')
            s = "find AVATA!"  # 替换为您的语音文本
        elif 'FrskyX20' in res:
            print('检测到FrskyX20')
            s = "find FrskyX20!"  # 替换为您的语音文本
        elif 'DJi' in res:
            print('检测到DJi')
            s = "find DJi!"  # 替换为您的语音文本'
        elif 'FrskyX20' in res:
            print('检测到Frsky')
            s = "find Frsky_X20!"  # 替换为您的语音文本
        elif 'Fubuta' in res:
            print('检测到Futaba_T14SG')
            s = "find Futaba_T14SG!"  # 替换为您的语音文本
        elif 'Inspire2' in res:
            print('检测到Inspire2')
            s = "find Inspire2!"  # 替换为您的语音文本
        elif 'Matrice100' in res:
            print('检测到Matrice100')
            s = "find Matrice100!"  # 替换为您的语音文本
        elif 'Matrice200' in res:
            print('检测到Matrice200')
            s = "find Matrice200!"  # 替换为您的语音文本
        elif 'Matrice600Pro' in res:
            print('检测到Matrice600Pro')
            s = "find Matrice600Pro!"  # 替换为您的语音文本
        elif 'Phantom3' in res:
            print('检测到Phantom3')
            s = "find Phantom3!"  # 替换为您的语音文本
        elif 'Phantom4PRO' in res:
            print('检测到Phantom4PRO')
            s = "find Phantom4PRO!"  # 替换为您的语音文本
        elif 'PhantomPRORTK' in res:
            print('检测到Phantom4PRORTK')
            s = "find Phantom4PRORTK!"  # 替换为您的语音文本
        elif 'Tarains_Plus' in res:
            print('Tarains_Plus')
            s = "find Tarains_Plus!"  # 替换为您的语音文本
        else:
            s = 'no Drone'

        with open(res_path + "result.txt", "w") as output_file:
            output_file.write(s)
        return 'Done'


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


if __name__ == '__main__':
    main()
