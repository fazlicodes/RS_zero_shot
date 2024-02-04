import torch

# Define the path to your pretrained model file
# model_path = '/home/mohamed.imam/Thesis/RS_zero_shot/all_weights/EVA02_CLIP_B_psz16_s8B.pt'
model_path = "/l/users/sanoojan.baliah/Felix/RS_zero_shot/svl_adapter_models/eurosat/eurosat_resnet50_georsclip.pt"

# Load the pretrained model
model = torch.load(model_path, map_location=torch.device('cpu'))

# Print all the keys in the model
print("Model Keys:")
print(model['results'])
# for key,value in model.items():
#     print(key)
