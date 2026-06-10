import torch
from math import ceil

if torch.cuda.is_available():
    print("CUDA is available. You can use GPU for computations.")
    print('CUDA Device Name:', torch.cuda.get_device_name(0))
    print('CUDA Device Count:', torch.cuda.device_count())
    print('CUDA Memory Capacity:', ceil(torch.cuda.get_device_properties(0).total_memory / (1024**3)), 'GB')
    print('CUDA Version:', torch.version.cuda)
else:
    print("CUDA is not available. You will be using CPU for computations.")