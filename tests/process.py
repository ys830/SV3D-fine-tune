# import os
# import json

# # 设置源文件夹和目标JSON文件路径
# source_folder = "/data/yisi/mywork/SV3D-fine-tune/objav_outputs_3/"
# output_json_path = "objav.json"

# def generate_json(source_folder, output_json_path):
#     # 获取源文件夹下所有以 .png 结尾的文件
#     image_files = [f for f in os.listdir(source_folder) if f.endswith('.png')]
#     image_files.sort()  # 对文件进行排序，确保重命名顺序正确

#     # 初始化路径列表
#     image_paths = []

#     # 遍历图像文件并重命名
#     for i, image_file in enumerate(image_files):
#         # 构建旧路径和新文件名
#         old_path = os.path.join(source_folder, image_file)
#         new_filename = f"{i:03d}.png"  # 格式化为四位数的文件名
#         new_path = os.path.join(source_folder, new_filename)

#         # 重命名文件
#         os.rename(old_path, new_path)

#         # 将新路径添加到列表
#         image_paths.append(new_path)

#     # 将图像路径列表写入JSON文件
#     with open(output_json_path, 'w') as json_file:
#         json.dump(image_paths, json_file, indent=4)

#     print(f"JSON file created at: {output_json_path}")

# # 调用函数生成JSON文件
# generate_json(source_folder, output_json_path)

import torch
import os

# 指定 .pt 文件的目录
pt_directory = "/data/yisi/mywork/SV3D-fine-tune/objav_outputs"

# 初始化一个空列表来存储加载的张量
all_latents = []

# 遍历目录中的所有文件，读取 .pt 文件
for filename in sorted(os.listdir(pt_directory)):
    if filename.endswith('.pt'):
        # 构造完整的文件路径
        file_path = os.path.join(pt_directory, filename)
        
        # 加载 .pt 文件中的张量
        latent = torch.load(file_path)
        
        # 将加载的张量添加到列表中
        all_latents.append(latent)

# 将所有的张量在第 0 个维度上合并
video_latent = torch.cat(all_latents, dim=0).detach()

# 将合并后的张量保存为 video_latent.pt
output_path = os.path.join(pt_directory, 'orbit_frame.pt')
torch.save(video_latent, output_path)

print(f"Saved combined latent tensor to {output_path} with shape {video_latent.shape}")

# import os
# import shutil

# # Define the source and destination directories
# source_dir = "/data/yisi/mywork/SV3D-fine-tune/objav_outputs"
# destination_dir = "/data/yisi/mywork/SV3D-fine-tune/objav_output2"

# # Ensure the destination directory exists
# os.makedirs(destination_dir, exist_ok=True)

# # Create folders and copy files with renaming as specified
# for i in range(21):  # Assuming we are creating folders 000-000 to 000-020
#     folder_name = f"000-{i:03d}"  # Format as 000-000, 000-001, ..., 000-020
#     folder_path = os.path.join(destination_dir, folder_name)
#     os.makedirs(folder_path, exist_ok=True)

#     # Define the file to copy and rename based on the index
#     pt_filename = f"{i:03d}.pt"
#     pt_source_path = os.path.join(source_dir, pt_filename)
#     pt_dest_path = os.path.join(folder_path, "orbit_frame.pt")

#     # Copy the .pt file if it exists
#     if os.path.exists(pt_source_path):
#         shutil.copy(pt_source_path, pt_dest_path)

#     # Copy 020.png into each folder
#     png_source_path = os.path.join(source_dir, "020.png")
#     png_dest_path = os.path.join(folder_path, "020.png")
    
#     if os.path.exists(png_source_path):
#         shutil.copy(png_source_path, png_dest_path)

# print("Folders created and files copied as specified.")
