import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from model.speech import file_path

# 加载音频文件
audio = AudioSegment.from_wav(file_path.audio_path_no_voice)

# 检测非静音部分（有人声的部分）
nonsilent_ranges = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-40)

# 使用SpeechRecognition库识别语音
recognizer = sr.Recognizer()

for start, end in nonsilent_ranges:
    segment = audio[start:end]
    segment.export("temp.wav", format="wav")
    with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print(f"识别到的语音: {text}")
        except sr.UnknownValueError:
            print("无法识别语音")
        except sr.RequestError as e:
            print(f"识别服务出错: {e}")
