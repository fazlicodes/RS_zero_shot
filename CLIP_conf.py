import os
from sklearn.model_selection import train_test_split
from torchvision.datasets import VisionDataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from torchvision import transforms

# Define the EuroSAT dataset class
class EuroSAT(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(EuroSAT, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split
        self.data = self.load_data()
        self.classes = list(set(self.data['label']))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def load_data(self):
        data_path = os.path.join(self.root, '2750')
        image_paths = []
        labels = []

        for label in os.listdir(data_path):
            label_path = os.path.join(data_path, label)
            for image_file in os.listdir(label_path):
                image_paths.append(os.path.join(label_path, image_file))
                labels.append(label)

        data = {'image_paths': image_paths, 'label': labels}
        return data

    def __len__(self):
        return len(self.data['image_paths'])

    def __getitem__(self, idx):
        img_path, target = self.data['image_paths'][idx], self.data['label'][idx]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

# Load the CLIP model and processor
model_name = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the transformation for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Function to compute the confidence score for a single image and text pair
def get_confidence_score(image_path, text):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    text = processor(text, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=image, text=text)
        logits_per_image = outputs.logits_per_image

    confidence_score = torch.softmax(logits_per_image, dim=1).squeeze().max().item()
    return confidence_score

# Function to compute the average confidence score for a dataset
def average_confidence_score(dataset):
    total_score = 0
    num_samples = len(dataset)

    for sample in dataset:
        image, text = sample
        confidence_score = get_confidence_score(image, text)
        total_score += confidence_score

    avg_score = total_score / num_samples
    return avg_score

# Example usage with EuroSAT dataset
root_path = '/path/to/eurosat_dataset'
eurosat_dataset = EuroSAT(root=root_path, transform=transform)

# Split the dataset into train and test sets
train_dataset, test_dataset = train_test_split(eurosat_dataset, test_size=0.2, random_state=42)

# Compute average confidence scores
avg_train_confidence = average_confidence_score(train_dataset)
avg_test_confidence = average_confidence_score(test_dataset)

print(f"Average Confidence Score (Train): {avg_train_confidence}")
print(f"Average Confidence Score (Test): {avg_test_confidence}")
