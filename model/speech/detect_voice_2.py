import librosa
import numpy as np

from model.speech import file_path

# 加载音频文件
audio_path = file_path.audio_file_path
y, sr = librosa.load(audio_path)

# 计算梅尔频率倒谱系数（MFCC）
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# 计算短时能量
frame_length = 2048
hop_length = 512
energy = np.array([
    sum(abs(y[i:i+frame_length]**2))
    for i in range(0, len(y), hop_length)
])

# 设置能量阈值来识别人声部分
energy_threshold = np.median(energy) * 1.5
voice_indices = np.where(energy > energy_threshold)[0]

# 将结果转换为时间
voice_times = librosa.frames_to_time(voice_indices, sr=sr, hop_length=hop_length)


print("人声部分的时间段（秒）：", voice_times)
print(len(voice_times))