import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# 要处理的根目录
input_root = '/data/yisi/mywork/SV3D-fine-tune/input_data'

def resize_and_add_alpha(image_path):
    # 打开图片
    image = Image.open(image_path).convert('RGB')  # 确保转换为RGB格式
    # 将图片调整到(576, 576)
    image = image.resize((576, 576), Image.LANCZOS)
    
    # 将图片转换为NumPy数组，并添加Alpha通道
    image_np = np.array(image)  # (576, 576, 3)
    alpha_channel = np.ones((576, 576), dtype=np.uint8) * 255  # 创建全为255的Alpha通道
    
    # 拼接RGB和Alpha通道，形成(576, 576, 4)的图像
    image_with_alpha = np.dstack((image_np, alpha_channel))
    
    # 转换回PIL的Image对象
    return Image.fromarray(image_with_alpha)

def process_folder(folder_path):
    # 获取所有.png文件
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    # 使用tqdm添加进度条
    for filename in tqdm(png_files, desc=f"Processing {folder_path}", unit="file"):
        image_path = os.path.join(folder_path, filename)
        # 调整图片大小并添加Alpha通道
        new_image = resize_and_add_alpha(image_path)
        # 保存图片，覆盖原有的文件
        new_image.save(image_path)

def process_all_folders(root_dir):
    # 遍历根目录下的所有文件夹
    folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for folder_name in folders:
        folder_path = os.path.join(root_dir, folder_name)
        process_folder(folder_path)

# 执行处理
process_all_folders(input_root)
