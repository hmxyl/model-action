import os

import cv2

from PIL import Image
import imagehash


class ImageDuplicateA:
    def extract_and_match_features(self, image1, image2):
        """
        图片判重（方法1）
        :param image1:
        :param image2:
        :return:
        """
        feature_detector = 'ORB'
        match_threshold = 0.95
        if feature_detector == 'ORB':
            detector = cv2.ORB_create()
        else:
            # 默认使用ORB
            detector = cv2.ORB_create()

        image1 = cv2.imread(image1)
        image2 = cv2.imread(image2)

        # 检测并计算描述符
        keypoints_1, descriptors1 = detector.detectAndCompute(image=image1, mask=None)
        keypoints_2, descriptors2 = detector.detectAndCompute(image=image2, mask=None)

        # 创建匹配器并进行匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        # 根据匹配结果判断是否相似
        if len(matches) / min(len(keypoints_1), len(keypoints_2)) > match_threshold:
            return True
        else:
            return False


class ImageDuplicateB:
    # 定义一个函数来计算图片的哈希值
    def calculate_image_hash(self, image_path):
        # 打开图片文件
        image = Image.open(image_path)
        # 使用Average Hash算法计算图片的哈希值
        hash_value = imagehash.average_hash(image)
        return hash_value

    def check_duplicate_image(self, image1_path, image2_path):
        """
        图片判重（方法2）
        :param image1_path:
        :param image2_path:
        :return:
        """
        # 计算图片的哈希值
        hash1 = self.calculate_image_hash(image1_path)
        hash2 = self.calculate_image_hash(image2_path)
        # 比较哈希值是否一致
        if hash1 == hash2:
            # 除了色彩之外，两张图片的内容完全一致。
            return True
        else:
            # "除了色彩之外，两张图片的内容不一致。
            return False


def check_duplicate(image_target, folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            image_path = os.path.join(root, file)
            if image_target != image_path and ImageDuplicateB().check_duplicate_image(image_target, image_path):
                return True


def remove_duplicate(folder_path):
    duplicate_file = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            image_target = os.path.join(root, file)
            if check_duplicate(image_target, folder_path):
                duplicate_file.append(image_target)
    for file in duplicate_file:
        os.remove(file)
