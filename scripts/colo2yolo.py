import os
import json
from tqdm import tqdm

def convert_coco_to_yolo(json_path, img_dir, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    cat2id = {name: idx for idx, name in enumerate(sorted(categories.values()))}

    print(f"类别映射（YOLO类ID → 原始类名）: {cat2id}")

    label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(label_dir, exist_ok=True)

    image_id_map = {img["id"]: img for img in data["images"]}

    for ann in tqdm(data["annotations"]):
        img = image_id_map[ann["image_id"]]
        img_w, img_h = img["width"], img["height"]
        file_name = img["file_name"]
        label_name = os.path.splitext(file_name)[0] + ".txt"

        x, y, w, h = ann["bbox"]
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w /= img_w
        h /= img_h
        class_name = categories[ann["category_id"]]
        class_id = cat2id[class_name]

        with open(os.path.join(label_dir, label_name), 'a') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print(f"✅ 标签已转换完成，保存在：{label_dir}")
    print(f"📌 共转换类别数：{len(cat2id)}")
    print(f"📌 类别顺序（YOLO编号）：{list(cat2id.keys())}")
    return list(cat2id.keys())

if __name__ == "__main__":
    json_path = "your_json_path_directory"
    img_dir = "your_images_directory"
    output_dir = "output_labels_directory"

    class_names = convert_coco_to_yolo(json_path, img_dir, output_dir)
    print("done")