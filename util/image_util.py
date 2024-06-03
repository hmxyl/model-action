import os

import cv2

SIMILARITY_THRESHOLD = 0.99


def extract_and_match_features(image1, image2):
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


def check_duplicate(image_target, folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            image_path = os.path.join(root, file)
            if image_target != image_path and extract_and_match_features(image_target, image_path):
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
