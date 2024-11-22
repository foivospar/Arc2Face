import os
from PIL import Image
import numpy as np

# 配置文件夹路径
ckpt=[1000,5000,10000,20000]
img_dir='sd-model-fine-tuned-new1'
img_dir2='sd-model-fine-tuned-new2'
ori_folders=['data/aligned/alignment/align_size(572,572)_move(0.250,0.000)_face_factor(0.450)/data',os.path.join(img_dir2,'output')]
folders=[os.path.join(img_dir,f'checkpoint-{str(i)}','output_ema') for i in ckpt]

output_image_path = "output_image_ema.jpg"  # 拼接后的输出路径

# 获取所有文件夹中的文件名（假设所有文件夹中的文件名相同）
file_names = os.listdir(folders[0])  # 获取第一个文件夹的所有文件名

# 初始化存储拼接后的图片
all_images = []
target_size=(512,512)
all_folders=ori_folders+folders
print(all_folders)


# 遍历每个文件名
for file_name in file_names[:10]:
    # 读取每个文件夹中该文件名的图片
    images = []
    for folder in all_folders:
        image_path = os.path.join(folder, file_name)
        if os.path.exists(image_path):
            image = Image.open(image_path)
            image = image.resize(target_size)
            images.append(image)
        else:
            print(f"Warning: {image_path} does not exist, skipping.")
    
    # 将每个文件夹的图片按列拼接
    if images:
        images_combined = np.concatenate([np.array(img) for img in images], axis=1)  # 按列拼接
        all_images.append(images_combined)

# 将所有拼接好的行按行拼接
if all_images:
    final_image = np.concatenate(all_images, axis=0)  # 按行拼接
    final_image = Image.fromarray(final_image)  # 转换回Image对象

    # 保存拼接后的图片
    final_image.save(output_image_path)
    print(f"Output image saved to {output_image_path}")
