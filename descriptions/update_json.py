import json

def update_json(input_file, output_file):
    # Read the original JSON data from the input file
    with open(input_file, 'r') as file:
        original_data = json.load(file)

    # Create a new dictionary with 'other' as the key and the values as a list
    new_data = {"other": []}

    # Merge the values from the original dictionary into the 'other' key
    for key, values in original_data.items():
        new_data["other"].extend(values)

    # Write the updated JSON data to the output file
    with open(output_file, 'w') as file:
        json.dump(new_data, file, indent=4)


import random
def sample_and_create_json(input_file, output_file):
    # Read the original JSON data from the input file
    with open(input_file, 'r') as file:
        original_data = json.load(file)

    # Sample 3 values from the 'other' key
    sampled_values = random.sample(original_data['other'], 3)

    # Create a new dictionary with 'sampled_data' as the key and the sampled values as a list
    new_data = {"other": sampled_values}

    # Write the sampled data to the output file
    with open(output_file, 'w') as file:
        json.dump(new_data, file, indent=4)


def merge_json_files(input_file, output_file, merged_file):
    # Read the data from the input file
    with open(input_file, 'r') as input_file:
        input_data = json.load(input_file)

    # Read the data from the output file
    with open(output_file, 'r') as output_file:
        output_data = json.load(output_file)

    # Merge the data into a new dictionary
    merged_data = {"input_data": input_data, "output_data": output_data}

    # Write the merged data to the new JSON file
    with open(merged_file, 'w') as merged_file:
        json.dump(merged_data, merged_file, indent=4)

# Example usage:
input_file_path = 'ImageNet.json'
output_file_path = 'ImageNet_other.json'
final_output_file_path = 'ImageNet_sampled.json'
merged_file_path = 'merged_data.json'


update_json(input_file_path, output_file_path)
sample_and_create_json(output_file_path, final_output_file_path)
merge_json_files("EuroSAT.json", final_output_file_path, merged_file_path)

