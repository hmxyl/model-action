import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from model.speech import file_path

if __name__ == '__main__':
    # Paraformer语音识别-中文-通用-16k-离线-large-长音频版
    # https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch

    """
    问题：No module named 'hdbscan'
    处理：pip install hdbscan
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model='iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch', model_revision="v2.0.4",
        vad_model='iic/speech_fsmn_vad_zh-cn-16k-common-pytorch', vad_model_revision="v2.0.4",
        punc_model='iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch', punc_model_revision="v2.0.3",
        spk_model="iic/speech_campplus_sv_zh-cn_16k-common", spk_model_revision="v2.0.2",
        device=device
    )

    rec_result = inference_pipeline(file_path.audio_file_path_3)
    print(rec_result)
