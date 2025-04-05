import os
import random
import shutil

def split(img_path: str, train_ratio: float = 0.8):
    """
    随机划分数据集为训练集和测试集，并将数据复制到新的文件夹中。

    :param img_path: 存储图像和标签的文件夹路径
    :param train_ratio: 训练集占比，默认为 80%
    :return: 返回训练集和测试集图像与标签路径列表
    """

    # 设置图像和标签的文件夹路径
    image_dir = os.path.join(img_path, 'images')
    label_dir = os.path.join(img_path, 'labels')

    # 获取所有图像和标签文件名
    images = os.listdir(image_dir)
    labels = os.listdir(label_dir)

    # 确保图像和标签数量一致
    if len(images) != len(labels):
        print(f"错误：图像和标签数量不匹配！图像数量: {len(images)}, 标签数量: {len(labels)}")
        return None, None

    # 检查图像文件和标签文件是否一一对应
    image_files = set([os.path.splitext(image)[0] for image in images])  # 去掉扩展名
    label_files = set([os.path.splitext(label)[0] for label in labels])  # 去掉扩展名

    # 检查是否存在缺失的标签文件或图像文件
    missing_labels = image_files - label_files
    missing_images = label_files - image_files

    if missing_labels:
        print(f"错误：以下图像没有对应的标签文件: {missing_labels}")
    if missing_images:
        print(f"错误：以下标签文件没有对应的图像: {missing_images}")

    if missing_labels or missing_images:
        return None, None

    # 随机打乱图像文件
    data = list(images)  # 图像文件名列表
    random.shuffle(data)

    # 划分训练集和测试集
    train_size = int(len(data) * train_ratio)
    train_images = data[:train_size]
    test_images = data[train_size:]

    # 生成训练集和测试集的标签路径
    train_labels = [f'{os.path.splitext(image)[0]}.txt' for image in train_images]
    test_labels = [f'{os.path.splitext(image)[0]}.txt' for image in test_images]

    # 创建训练集和测试集文件夹
    train_image_dir = os.path.join(img_path, 'train', 'images')
    test_image_dir = os.path.join(img_path, 'test', 'images')
    train_label_dir = os.path.join(img_path, 'train', 'labels')
    test_label_dir = os.path.join(img_path, 'test', 'labels')

    # 如果没有目标文件夹，创建它们
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    # 将文件复制到相应的文件夹
    for img, lbl in zip(train_images, train_labels):
        shutil.copy(os.path.join(image_dir, img), os.path.join(train_image_dir, img))
        shutil.copy(os.path.join(label_dir, lbl), os.path.join(train_label_dir, lbl))

    for img, lbl in zip(test_images, test_labels):
        shutil.copy(os.path.join(image_dir, img), os.path.join(test_image_dir, img))
        shutil.copy(os.path.join(label_dir, lbl), os.path.join(test_label_dir, lbl))

    # 返回训练集和测试集的图像和标签路径
    train_image_paths = [os.path.join(train_image_dir, img) for img in train_images]
    test_image_paths = [os.path.join(test_image_dir, img) for img in test_images]
    train_label_paths = [os.path.join(train_label_dir, lbl) for lbl in train_labels]
    test_label_paths = [os.path.join(test_label_dir, lbl) for lbl in test_labels]





if __name__=="__main__":
    img_path = "D:\\code\\python\\Outsourcing\\captcha\\QK_CAPTCHA"  # 假设数据集存储在这个路径下
    split(img_path)
