import os
import os

def print_folder_names(directory):
    names=[]
    for entry in os.scandir(directory):
        if entry.is_dir():
            # print(entry.name)
            names.append(entry.name)
    print(sorted(names))
    with open('names.txt','w') as f:
        for i in sorted(names):
            f.write(f'"{i}":"{i.replace("_"," ")}",\n')
# print_folder_names("data/resisc45/NWPU-RESISC45/NWPU-RESISC45/")

import json

def update_json_keys(input_file, output_file):
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    updated_data = {}
    for key, value in data.items():
        updated_key = key.replace(' ', '_')
        updated_data[updated_key] = value
    
    with open(output_file, 'w') as file:
        json.dump(updated_data, file, indent=4)

# Replace 'input.json' with your JSON file and 'output.json' with the desired output file name
update_json_keys('./descriptions/generic/RESISC45.json', 'output.json')
