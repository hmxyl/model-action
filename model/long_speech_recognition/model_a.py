import torch
from funasr import AutoModel

from model.speech import file_path
if __name__ == '__main__':
    """
    语音识别: https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary
    """

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # paraformer-zh is a multi-functional asr model
    # use vad, punc, spk or not as you need
    model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                      vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                      punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                      # spk_model="cam++", spk_model_revision="v2.0.4",
                      device=device
                      )
    res = model.generate(input=file_path.audio_file_path,
                         batch_size_s=300,
                         hotword='魔搭')
    print(res[0]['text'])
    # print(res)