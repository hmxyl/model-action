import json
import os
import shutil
import unittest

from model.speech import file_path
from util import detect_util
from model.ppt_detect.util.detect_util import DetestUtil

frame_fragment_file = 'D:\\Workspace\\test_model\\ppt\\课程资源.txt'
ppt_test_right_folder = 'D:\\Workspace\\test_model\\ppt\\课程资源_PPT\\'

# 静态帧判定阈值
STATIC_FRAME_THRESHOLD = detect_util.STATIC_FRAME_THRESHOLD
# 判定视频包含PPT，视频静态帧需高于的百分比
FRAGMENT_STATIC_THRESHOLD = detect_util.FRAGMENT_STATIC_THRESHOLD


def getFrameFragment():
    frame_fragment = []
    with open(frame_fragment_file, 'r') as file:
        for line in file:
            frame_fragment.append(json.loads(line))
    return frame_fragment


def getPptTestRight():
    ppt_test_right = []
    for root, dirs, files in os.walk(ppt_test_right_folder):
        for dir_name in dirs:
            ppt_test_right.append(int(dir_name))
    return sorted(ppt_test_right)


def getStaticFramePercent(fragment):
    frame_diffs = [frame['diff'] for frame in fragment][1:]
    static_frames = [diff for diff in frame_diffs if diff < STATIC_FRAME_THRESHOLD]
    return len(static_frames) / len(frame_diffs)


class FrameTest(unittest.TestCase):
    def test_show_analyze_right(self):
        frame_fragment = getFrameFragment()
        ppt_test_right = getPptTestRight()
        for fragment in frame_fragment:
            start_num = fragment[0]['num']
            static_percent = getStaticFramePercent(fragment)
            if start_num in ppt_test_right and static_percent < FRAGMENT_STATIC_THRESHOLD:
                print(str(start_num) + " ---------------- " + str(static_percent))
        print()
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print()
        for fragment in frame_fragment:
            start_num = fragment[0]['num']
            static_percent = getStaticFramePercent(fragment)
            if start_num not in ppt_test_right and static_percent > FRAGMENT_STATIC_THRESHOLD:
                print(str(start_num) + " ---------------- " + str(static_percent))

    def test_get_ppt_frame(self):
        video_path = file_path.ppt_file_path
        ppt_fragment = getFrameFragment()
        ppt_frame_num = DetestUtil().get_ppt_frame_num(ppt_fragment, video_path)
        # 保存图片
        picture_folder = 'D:\\Workspace\\test_model\\ppt\\课程资源\\'
        if os.path.exists(picture_folder):
            shutil.rmtree(picture_folder)
        if not os.path.exists(picture_folder):
            os.makedirs(picture_folder)
        DetestUtil().save_frame_picture(video_path, picture_folder, ppt_frame_num)

