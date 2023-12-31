import torch

# Assuming you have two tensors a and b with shape [50, 10]
a = torch.rand((50, 10))
b = torch.rand((50, 10))

# Combine the tensors along a new dimension (e.g., concatenate along a new dimension)
combined_tensor = torch.stack([a, b], dim=2)

# Average along the new dimension
average_tensor = torch.mean(combined_tensor, dim=2)

# Resulting tensor with shape [50, 10]
print(average_tensor.shape)
