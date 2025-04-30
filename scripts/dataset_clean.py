import os
import cv2

def clean_invalid_labels(label_dir):
    """
    检查标签文件中坐标是否超出 [0, 1] 范围，删除无效标签
    """
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        
        # 读取标签文件
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        valid_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # 跳过格式不对的行

            # 获取坐标信息
            x_center, y_center, width, height = map(float, parts[1:])
            
            # 检查坐标是否超出范围
            if all(0 <= val <= 1 for val in [x_center, y_center, width, height]):
                valid_lines.append(line)
            else:
                print(f"Skipping invalid label: {label_file}")
        
        # 将有效标签写回
        with open(label_path, 'w') as f:
            f.writelines(valid_lines)

def remove_invalid_images(image_dir, label_dir):
    """
    删除无标签或无法读取的图像
    """
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]  # 根据你的图像格式修改
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    # 获取图像文件名和标签文件名（去除扩展名）
    image_files = [os.path.splitext(f)[0] for f in image_files]
    label_files = [os.path.splitext(f)[0] for f in label_files]

    for image in image_files:
        image_path = os.path.join(image_dir, f"{image}.jpg")

        # 如果没有对应的标签，删除图像
        if image not in label_files:
            print(f"Deleting image without label: {image_path}")
            try:
                os.remove(image_path)
            except FileNotFoundError:
                print(f"File not found: {image_path}")
        
        # 检查图像是否能成功读取
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Deleting corrupt image: {image}.jpg")
                os.remove(image_path)
        except Exception as e:
            print(f"Error reading image: {image}.jpg, {e}")
            try:
                os.remove(image_path)
            except FileNotFoundError:
                print(f"File not found: {image}.jpg")

def main():
    # 图像和标签文件夹路径
    image_dir = 'your_new_images_directory'
    label_dir = 'your_new_labels_directory'
    
    # 清理无效标签
    print("\nCleaning invalid labels...")
    clean_invalid_labels(label_dir)

    # 清理无效图像
    print("\nRemoving invalid images...")
    remove_invalid_images(image_dir, label_dir)

    print("\nData cleaning completed.")

if __name__ == "__main__":
    main()
