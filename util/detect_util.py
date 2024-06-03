import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

from common.process_decorator import ProcessDecorator

# 场景变化判定阈值
FRAGMENT_CHANGE_THRESHOLD = 0.9
# 静态帧判定阈值
STATIC_FRAME_THRESHOLD = 0.02
# 判定视频包含PPT，视频静态帧需高于的百分比
VIDEO_STATIC_THRESHOLD = 0.5
# 判定视频包含PPT，视频静态帧需高于的百分比
FRAGMENT_STATIC_THRESHOLD = 0.9


class DetestUtil:
    @ProcessDecorator.tag("获取视频帧率和帧变化数据")
    def get_video_frame_diff(self, video_path):
        cap = cv2.VideoCapture(video_path)
        # 检查视频是否成功打开
        if not cap.isOpened():
            raise Exception('Failed to read video')
        ret, first_frame = cap.read()
        if not ret:
            return None
        # 帧差异数据
        frame_diffs = []
        # 转换帧为灰度图像
        prev_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(frame_gray, prev_frame_gray)
            non_zero_count = np.count_nonzero(diff)
            frame_diff = non_zero_count / diff.size
            frame_diffs.append(frame_diff)
            prev_frame_gray = frame_gray
        cap.release()
        return frame_diffs

    def check_contain_ppt(self, frame_diffs):
        if frame_diffs is None or len(frame_diffs) == 0:
            return False
        static_frames = [diff for diff in frame_diffs if diff < STATIC_FRAME_THRESHOLD]
        # 假定超过50%的帧是静态的， 则视频包含PPT
        if len(static_frames) / len(frame_diffs) > VIDEO_STATIC_THRESHOLD:
            return True
        return False

    @ProcessDecorator.tag("保存指定帧号的图片")
    def save_frame_picture(self, video_path, picture_folder, frame_num_group):
        if frame_num_group is None or len(frame_num_group) == 0:
            return
        cap = cv2.VideoCapture(video_path)
        # 检查视频是否成功打开
        if not cap.isOpened():
            raise Exception('Failed to read video')
        # 初始化帧计数器
        frame_num = 0
        # 逐帧读取视频并保存图片
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_num > frame_num_group[-1]:
                break
            if frame_num in frame_num_group:
                # 构建图片文件名
                frame_filename = os.path.join(picture_folder, f'{frame_num}.jpg')
                # 保存帧作为图片
                cv2.imwrite(frame_filename, frame)
            frame_num += 1
            # 关闭视频文件
        cap.release()

    def get_frame_fragment(self, frame_diffs):
        # 视频帧分片段
        frame_fragment = []
        fragment_temp = []
        for index, frame_gray in enumerate(frame_diffs):
            frame_data = {
                'num': index + 1,
                'diff': frame_gray,
            }
            if frame_gray > FRAGMENT_CHANGE_THRESHOLD:
                frame_fragment.append(fragment_temp)
                fragment_temp = [frame_data]
            else:
                fragment_temp.append(frame_data)
        # 获取包含PPT的场景片段
        ppt_fragment = []
        for fragment in frame_fragment:
            fragment_frame_diffs = [frame['diff'] for frame in fragment][1:]
            static_frames = [diff for diff in fragment_frame_diffs if diff < STATIC_FRAME_THRESHOLD]
            static_percent = len(static_frames) / len(fragment_frame_diffs)
            if static_percent > FRAGMENT_STATIC_THRESHOLD:
                ppt_fragment.append(fragment)
        return frame_fragment, ppt_fragment

    def __check_exist(self, target, frame_gray_group):
        if len(frame_gray_group) == 0:
            return False
        for frame_gray in frame_gray_group:
            # 计算当前帧和前一帧的相似度
            score, _ = compare_ssim(target, frame_gray, full=True)
            print(score)
            if score > (1 - STATIC_FRAME_THRESHOLD):
                # 相似度低于阈值，认为是新的帧，保存
                return True
        return False

    @ProcessDecorator.tag("帧去重", raise_exception=False)
    def __frame_deduplication(self, video_path, ppt_frame_num):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception('Failed to read video')

        frame_gray_group = []
        ppt_frame_num_clean = []
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_num > ppt_frame_num[-1]:
                break
            if frame_num in ppt_frame_num:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if self.__check_exist(frame_gray, frame_gray_group):
                    print('【重复帧】：' + str(frame_num))
                else:
                    ppt_frame_num_clean.append(frame_num)
                    frame_gray_group.append(frame_gray)
            frame_num += 1
        cap.release()
        return ppt_frame_num_clean

    def get_ppt_frame_num(self, ppt_fragment, video_path):
        # 待提取图片帧号
        ppt_frame_num = []
        for fragment in ppt_fragment:
            ppt_frame_num.append(fragment[0]['num'])
            for frame in fragment[1:]:
                if frame['diff'] > STATIC_FRAME_THRESHOLD:
                    ppt_frame_num.append(frame['num'])
        # 帧去重后返回
        return self.__frame_deduplication(video_path, ppt_frame_num)
