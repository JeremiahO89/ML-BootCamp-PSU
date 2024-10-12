import torch
print(torch.__version__)  # Should show version like 2.4.1+cu124
print(torch.version.cuda)  # Should show '12.4'
print(torch.cuda.is_available())  # Should return True
