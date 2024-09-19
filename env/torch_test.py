import os

import torch

'''
cuda12.5安装 pytorch： https://pytorch.org/get-started/previous-versions/
卸载：pip uninstall torch torchvision torchaudio
安装：pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
'''
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

default_home = "D:\\model_scope"

HF_HOME = os.path.expanduser(
    os.getenv(
        "HF_HOME",
        os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "huggingface"),
    )
)
print(HF_HOME)
print("-------------------------------")

print(os.getenv("HF_HOME"))
print(os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "huggingface"))
print(os.getenv("XDG_CACHE_HOME"))
print(os.getenv(
    "HF_HOME",
    os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "huggingface"),
))

