import torch

print('Checking device')
mul_device = torch.device('cpu')
if torch.cuda.is_available():
    mul_device = torch.device('cuda')
