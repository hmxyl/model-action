
import torch

'''
cuda12.5安装 pytorch： https://pytorch.org/get-started/previous-versions/
卸载：pip uninstall torch torchvision torchaudio
安装：pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
'''
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
