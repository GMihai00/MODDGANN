import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print(torch.cuda.current_device())  # Check the current CUDA device
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # Check the device name
print(torch.version.cuda)  # Check the CUDA version used by PyTorch

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())



