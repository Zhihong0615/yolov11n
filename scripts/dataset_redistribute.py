import os
import random
import shutil

def split_and_rename_data(img_dir, label_dir, output_train_img_dir, output_train_label_dir, output_val_img_dir, output_val_label_dir, train_ratio=0.8):
    # 创建输出目录
    os.makedirs(output_train_img_dir, exist_ok=True)
    os.makedirs(output_train_label_dir, exist_ok=True)
    os.makedirs(output_val_img_dir, exist_ok=True)
    os.makedirs(output_val_label_dir, exist_ok=True)

    # 获取图像和标签的文件列表
    images = sorted(os.listdir(img_dir))
    labels = sorted(os.listdir(label_dir))

    # 确保每个图像都有对应的标签
    assert len(images) == len(labels), "Number of images and labels don't match!"

    # 划分数据集：按照 8:2 的比例
    total_files = len(images)
    train_size = int(total_files * train_ratio)
    val_size = total_files - train_size

    # 打乱数据，确保划分的随机性
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)

    # 训练集和验证集的文件分配
    train_images, val_images = images[:train_size], images[train_size:]
    train_labels, val_labels = labels[:train_size], labels[train_size:]

    # 重新命名并复制训练集图像和标签
    for i, (img, lbl) in enumerate(zip(train_images, train_labels)):
        new_name = f"{i + 1:04d}"  # 格式化为四位数字
        img_src = os.path.join(img_dir, img)
        img_dst = os.path.join(output_train_img_dir, f"{new_name}.jpg")
        lbl_src = os.path.join(label_dir, lbl)
        lbl_dst = os.path.join(output_train_label_dir, f"{new_name}.txt")
        shutil.copy(img_src, img_dst)
        shutil.copy(lbl_src, lbl_dst)

    # 重新命名并复制验证集图像和标签
    for i, (img, lbl) in enumerate(zip(val_images, val_labels)):
        new_name = f"{i + 1 + len(train_images):04d}"  # 确保验证集命名从训练集之后继续
        img_src = os.path.join(img_dir, img)
        img_dst = os.path.join(output_val_img_dir, f"{new_name}.jpg")
        lbl_src = os.path.join(label_dir, lbl)
        lbl_dst = os.path.join(output_val_label_dir, f"{new_name}.txt")
        shutil.copy(img_src, img_dst)
        shutil.copy(lbl_src, lbl_dst)

    print("数据集划分和重命名完成！")

# 输入路径（图像和标签文件夹路径）
img_dir = r"your_images_directory(haven't been redistributed, which means the whole images)"
label_dir = r"your_labels_directory(haven't been redistributed, which means the whole labels)"

# 输出路径（新的训练集和验证集文件夹路径）
output_train_img_dir = r"output_train_images_directory"
output_train_label_dir = r"your_train_labels_directory"
output_val_img_dir = r"your_val_images_directory"
output_val_label_dir = r"your_val_labels_directory"

# 调用函数划分数据集并重命名
split_and_rename_data(img_dir, label_dir, output_train_img_dir, output_train_label_dir, output_val_img_dir, output_val_label_dir, train_ratio=0.8)
