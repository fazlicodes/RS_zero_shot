import torch

# Assuming you have tensor1 and tensor2 with the same shape [50, 10]
tensor1 = torch.rand((5, 3))
tensor2 = torch.rand((5, 3))

# Choose a value for alpha in the range [0, 1]
alpha = 0.2

# Combine the tensors based on the weighted formula
combined_tensor = alpha * tensor1 + (1 - alpha) * tensor2

# Resulting tensor with the same shape [50, 10]
print(combined_tensor)
