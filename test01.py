import torch

def split_other_tensor(other_tensor, group_tensor, num_groups):
    # Map unique string values to numerical indices
    unique_groups, group_indices = torch.unique(group_tensor, return_inverse=True)
    
    # Get the total number of unique groups
    total_groups = len(unique_groups)

    # Ensure the number of specified groups is valid
    if num_groups > total_groups:
        raise ValueError("The specified number of groups is greater than the total number of unique groups.")

    # Split the indices into the specified number of groups
    group_indices_split = torch.chunk(torch.arange(total_groups), num_groups)

    # Create a list to store tensors for each group
    group_list = []

    # Iterate over each group and create tensors for the "other" tensor
    for indices in group_indices_split:
        subset_other_tensor = other_tensor[group_indices == indices[0]]  # Assuming group indices are consistent
        group_list.append(subset_other_tensor)

    # Stack the tensors to create a tensor of shape (num_groups, group_dim, emb_dim)
    other_tensor_split = torch.stack(group_list)

    return other_tensor_split

# Example usage:
group_dim_tensor = torch.tensor([1,1,2,2,3,3]) # Using numerical indices instead of strings
other_tensor = torch.rand((6, 4))  # Example tensor with dimensions (N, emb_dim)

result = split_other_tensor(other_tensor, group_dim_tensor, num_groups=2)

print(f"Split Other Tensor:\n{result}\n{'='*30}")
print(result.shape)
