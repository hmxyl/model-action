from model.speech import file_path

if __name__ == '__main__':
    # https://www.modelscope.cn/models/iic/speech_campplus_speaker-diarization_common
    # 版本要求 modelscope version 升级至最新版本 funasr 升级至最新版本
    from modelscope.pipelines import pipeline

    sd_pipeline = pipeline(
        task='speaker-diarization',
        model='damo/speech_campplus_speaker-diarization_common',
        model_revision='v1.0.0'
    )
    result = sd_pipeline(file_path.audio_file_path)
    print(result)
    print("[重新设置后]")
    # 如果有先验信息，输入实际的说话人数，会得到更准确的预测结果
    result = sd_pipeline(file_path.audio_file_path, oracle_num=2)
    print(result)
