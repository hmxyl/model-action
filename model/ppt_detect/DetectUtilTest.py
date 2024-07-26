import json
import os
import shutil
import unittest

from model.ppt_detect.util.detect_util import DetestUtil
from model.speech import file_path
from util import image_util


def check_ppt_frame_fragment(ppt_fragment):
    # 根据test_frame_fragment结果手动分析保留的包含PPT文件夹
    ppt_test_right_folder = 'D:\\Workspace\\test_model\\ppt\\课程资源_PPT\\'
    ppt_test_right = []
    for root, dirs, files in os.walk(ppt_test_right_folder):
        for dir_name in dirs:
            ppt_test_right.append(int(dir_name))
    ppt_test_right = sorted(ppt_test_right)
    # 校验
    ppt_analyze = [fragment[0]['num'] for fragment in ppt_fragment]
    print("-------不正确的PPT片段-------------------------------")
    for start_num in ppt_analyze:
        if start_num not in ppt_test_right:
            print(start_num)
    print("-------丢失的PPT片段-------------------------------")
    for start_num in ppt_test_right:
        if start_num not in ppt_analyze:
            print(start_num)


video_path = file_path.ppt_file_path
picture_folder = file_path.ppt_picture_folder
picture_all_folder = file_path.ppt_picture_all_folder


class FrameTest(unittest.TestCase):
    def test_check_exist_ppt(self):
        """
        判断视频是否包含PPT
        """
        frame_diffs = DetestUtil().get_video_frame_diff(file_path.ppt_file_path)
        contains_ppt = DetestUtil().check_contain_ppt(frame_diffs)
        print("视频是否包含PPT:", contains_ppt)

    def test_frame_fragment(self):
        """
        视频按照帧变化进行拆分片段（all）
        """
        frame_fragment_file = 'D:\\Workspace\\test_model\\ppt\\课程资源.txt'
        frame_diffs = DetestUtil().get_video_frame_diff(video_path)
        frame_fragment, ppt_fragment = DetestUtil().get_frame_fragment(frame_diffs)
        # 保存片段数据到文件
        os.remove(frame_fragment_file)
        with open(frame_fragment_file, 'w') as file:
            for fragment in ppt_fragment:
                file.write(json.dumps(fragment))
                file.write("\n")
        # 按片段保存每帧视频
        for fragment in frame_fragment:
            fragment_picture_folder = os.path.join(picture_all_folder, str(fragment[0]['num']))
            frame_num_group = [frame['num'] for frame in fragment]
            if os.path.exists(fragment_picture_folder):
                shutil.rmtree(fragment_picture_folder)
            if not os.path.exists(fragment_picture_folder):
                os.makedirs(fragment_picture_folder)
            DetestUtil().save_frame_picture(video_path, fragment_picture_folder, frame_num_group)

    def test_ppt_detect(self):
        # 分析视频
        frame_diffs = DetestUtil().get_video_frame_diff(video_path)
        frame_fragment, ppt_fragment = DetestUtil().get_frame_fragment(frame_diffs)
        ppt_frame_num = DetestUtil().get_ppt_frame_num(ppt_fragment, video_path)
        # 验证
        # check_ppt_frame_fragment(ppt_fragment)
        # 提取PPT图片
        if os.path.exists(picture_folder):
            shutil.rmtree(picture_folder)
        if not os.path.exists(picture_folder):
            os.makedirs(picture_folder)
        DetestUtil().save_frame_picture(video_path, picture_folder, ppt_frame_num)
        # 图片去重
        # image_util.remove_duplicate(picture_folder)

    def test_remove_duplicate(self):
        # 图片去重
        image_util.remove_duplicate(picture_folder)
