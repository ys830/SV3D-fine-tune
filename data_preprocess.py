import os
import glob
import json
import collections
import numpy as np
import shutil
import torch

def process_directory(root_dir):
    # 1. 将images文件夹中的所有图片上提一级并重命名
    images_dir = os.path.join(root_dir, 'images')
    image_paths = sorted(glob.glob(os.path.join(images_dir, '*.png')))
    
    for idx, img_path in enumerate(image_paths):
        # 新的文件名为000.png, 001.png...
        new_img_name = f"{str(idx).zfill(3)}.png"
        new_img_path = os.path.join(root_dir, new_img_name)
        shutil.move(img_path, new_img_path)  # 移动并重命名文件

    # 删除images文件夹
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)

    # 2. 读取transforms_train.json，提取3x4矩阵并保存为.npy文件
    json_path = os.path.join(root_dir, 'transforms_train.json')
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            meta = json.load(f, object_pairs_hook=collections.OrderedDict)
        
        poses = []
        for idx, frame in enumerate(meta['frames']):
            pose = np.array(frame['transform_matrix'])[:3, :4]
            poses.append(pose)
            # 保存3x4矩阵为.npy文件
            npy_path = os.path.join(root_dir, f"{str(idx).zfill(3)}.npy")
            np.save(npy_path, pose)
        
        # 删除json文件
        os.remove(json_path)

    # 3. 生成cardiac.json文件，保存所有图片的路径
    all_image_paths = sorted(glob.glob(os.path.join(root_dir, '*.png')))
    
    cardiac_json_path = os.path.join(root_dir, 'cardiac.json')
    with open(cardiac_json_path, 'w') as f:
        json.dump(all_image_paths, f, indent=4)

def main():
    base_dir = '/data/yisi/mywork/SV3D-fine-tune/input_data'
    sub_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for sub_dir in sub_dirs:
        process_directory(os.path.join(base_dir, sub_dir))

if __name__ == "__main__":
    main()
