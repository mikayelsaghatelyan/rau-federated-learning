import torch
print(torch.cuda.is_available())  # Should return True if GPU is detected
print(torch.cuda.device_count())  # Number of GPUs available
print(torch.cuda.get_device_name(0))  # Name of the first GPU
print(torch.version.cuda)  # CUDA version PyTorch is using
print(torch.backends.cudnn.version())  # cuDNN version