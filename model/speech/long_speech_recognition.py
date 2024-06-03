from faster_whisper import WhisperModel
from funasr import AutoModel

from model.speech import file_path


def test_speech_long_recognition():
    """
     whisper 语音识别
     HF_ENDPOINT=https://hf-mirror.com
    """

    model = WhisperModel("large-v2")
    segments, info = model.transcribe(file_path.audio_file_path)
    for segment in segments:
        print("[%.3fs -> %.3fs] %s" % (segment.start, segment.end, segment.text))


def test_paraformer():
    """
    语音识别
    """
    # paraformer-zh is a multi-functional asr model
    # use vad, punc, spk or not as you need
    model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                      vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                      punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                      spk_model="cam++", spk_model_revision="v2.0.2",
                      )
    res = model.generate(input=file_path.audio_file_path,
                         batch_size_s=300,
                         hotword='魔搭')
    print(res[0]['text'])
    # print(res)


if __name__ == '__main__':
    """
    Whisper语音识别
    """
    test_speech_long_recognition()
    """
    paraformer-zh语音识别
    """
    # test_paraformer()
