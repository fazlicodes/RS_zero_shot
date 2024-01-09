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

# Example usage:
input_file_path = 'ImageNet.json'
output_file_path = 'ImageNet_other.json'

update_json(input_file_path, output_file_path)
