import struct
filepath = 'E:/Drone_dataset/雷达设备专采/圆/20240322164025.Bin'
# 打开二进制文件
with open(filepath, "rb") as f:
    # 读取帧头的前18个字节
    frame_header_bytes = f.read(18)

# 使用struct.unpack解析帧头数据
# '18s' 表示解析18字节的字符串
# '<' 表示小端字节序
frame_header = struct.unpack('<18s', frame_header_bytes)

# 解析结果是一个元组，其中的第一个元素就是帧头的内容
frame_header_content = frame_header[0]

# 打印帧头内容
print(frame_header_content)
