import time

import torch
from pyannote.audio import Pipeline

from model.speech import file_path

if __name__ == '__main__':
    # https://huggingface.co/pyannote/speaker-diarization-3.1
    time1 = time.time()
    pipeline = Pipeline.from_pretrained(
        checkpoint_path="pyannote/speaker-diarization-3.1",
        use_auth_token="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"
    )
    pipeline.to(torch.device("cuda:0"))
    # run the pipeline on an audio file
    diarization = pipeline(file_path.audio_file_path)
    print(diarization)
    time2 = time.time()
    print(time2 - time1)
    # print("[重新设置后]")
    # diarization = pipeline(file_path.audio_file_path, min_speakers=2, max_speakers=5)
    # print(diarization)
    # time3 = time.time()
    # print(time3 - time2)
