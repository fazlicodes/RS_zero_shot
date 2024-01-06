import torch

def split_into_groups(group_tensor, *other_tensors, num_groups):
    # Get unique groups and their indices
    unique_groups, group_indices = group_tensor.unique(return_inverse=True)

    # Get the total number of unique groups
    total_groups = len(unique_groups)

    # Ensure the number of specified groups is valid
    if num_groups > total_groups:
        raise ValueError("The specified number of groups is greater than the total number of unique groups.")

    # Split the indices into the specified number of groups
    group_indices_split = torch.chunk(torch.arange(total_groups), num_groups)

    # Create a list to store tensors for each group
    group_list = []

    # Iterate over each group and concatenate the tensors
    for indices in group_indices_split:
        subset_tensors = [tensor[group_indices == idx] for tensor in (group_tensor, *other_tensors)]
        group_list.append(tuple(subset_tensors))

    return group_list

# Example usage:
# data = {'group_dim': ['A', 'A', 'B', 'B', 'A', 'B'],
#         'other_column': [1, 2, 3, 4, 5, 6]}

group_dim_tensor = torch.random.choice(['A', 'B'], size=(6,))
other_column_tensor = torch.rand(6)

result = split_into_groups(group_dim_tensor, other_column_tensor, num_groups=2)

for i, group in enumerate(result):
    print(f"Group {i+1}:\n{group}\n{'='*30}")