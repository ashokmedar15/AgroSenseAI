import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
from torchvision import models , transforms , io
from sklearn.metrics import confusion_matrix , precision_recall_fscore_support
import time
from tqdm import tqdm

# Dataset paths and definitions
base_path = r"C:\Users\91905\PycharmProjects\AgroSense_AI_v3\datasets\AgroSense"
crops = ["apple" , "potato" , "tomato"]
pests = {
    "apple": ["Apple_Anthracnose" , "Apple_Aphids" , "Apple_Fruit_Fly" , "Apple_Powdery_Mildew"] ,
    "potato": ["Potato_Beetle" , "Potato_Tomato_Late_Blight" , "Potato_Early_blight"] ,
    "tomato": ["Tomato_Powdery_Mildew" , "Tomato_Aphids" , "Tomato_Leaf_Curl_Virus" , "Tomato_Late_blight"]
}
healthy = {
    "apple": ["Apple___healthy"] ,
    "potato": ["Potato___healthy"] ,
    "tomato": ["Tomato___healthy"]
}
diseases = {
    "apple": ["Apple___Apple_scab" , "Apple___Black_rot" , "Apple___Cedar_apple_rust"] ,
    "potato": ["Potato___Early_blight" , "Potato___Late_blight"] ,
    "tomato": ["Tomato___Bacterial_spot" , "Tomato___Early_blight" , "Tomato___Late_blight" ,
               "Tomato___Leaf_Mold" , "Tomato___Septoria_leaf_spot" , "Tomato___Target_Spot" ,
               "Tomato___Tomato_mosaic_virus" , "Tomato___Tomato_Yellow_Leaf_Curl_Virus"]
}

# Pest folder mapping
pest_folder_mapping = {
    "Apple_Anthracnose": "Anthracnose" ,
    "Apple_Aphids": "Aphids" ,
    "Apple_Fruit_Fly": "Fruit_Fly" ,
    "Apple_Powdery_Mildew": "Powdery_Mildew" ,
    "Potato_Beetle": "Potato_Beetle" ,
    "Potato_Tomato_Late_Blight": "Tomato_Late_Blight" ,
    "Potato_Early_blight": "Potato_Early_blight" ,
    "Tomato_Powdery_Mildew": "Powdery_Mildew" ,
    "Tomato_Aphids": "Aphids" ,
    "Tomato_Leaf_Curl_Virus": "Tomato_Leaf_Curl_Virus" ,
    "Tomato_Late_blight": "Tomato_Late_blight"
}

# Class mapping
class_mapping = {
    "Apple___healthy": 0 ,
    "Apple___Apple_scab": 1 ,
    "Apple___Black_rot": 2 ,
    "Apple___Cedar_apple_rust": 3 ,
    "Apple_Anthracnose": 4 ,
    "Apple_Aphids": 5 ,
    "Apple_Fruit_Fly": 6 ,
    "Apple_Powdery_Mildew": 7 ,
    "Potato___healthy": 8 ,
    "Potato___Early_blight": 9 ,
    "Potato___Late_blight": 10 ,
    "Potato_Beetle": 11 ,
    "Potato_Tomato_Late_Blight": 12 ,
    "Potato_Early_blight": 13 ,
    "Tomato___healthy": 14 ,
    "Tomato___Bacterial_spot": 15 ,
    "Tomato___Early_blight": 16 ,
    "Tomato___Late_blight": 17 ,
    "Tomato___Leaf_Mold": 18 ,
    "Tomato___Septoria_leaf_spot": 19 ,
    "Tomato___Target_Spot": 20 ,
    "Tomato___Tomato_mosaic_virus": 21 ,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 22 ,
    "Tomato_Powdery_Mildew": 23 ,
    "Tomato_Aphids": 24 ,
    "Tomato_Leaf_Curl_Virus": 25 ,
    "Tomato_Late_blight": 26
}
class_names = list(class_mapping.keys())

# Pre-load metadata
metadata_dict = {}


def preload_metadata():
    for crop in crops:
        for disease in diseases[crop]:
            metadata = {"causes": "" , "prevention": "" , "recommendations": ""}
            base_path_meta = os.path.join(base_path , crop , "train" , "crops" , "diseased" , disease)
            for key in metadata:
                possible_files = [f"{key}.txt" , f"{key}s.txt"] if key == "prevention" else [f"{key}.txt"]
                for file_name in possible_files:
                    file_path = os.path.join(base_path_meta , file_name)
                    if os.path.exists(file_path):
                        with open(file_path , "r" , encoding="utf-8") as f:
                            metadata[key] = f.read().strip()
                        break
            metadata_dict[disease] = metadata

        for pest in pests[crop]:
            metadata = {"causes": "" , "prevention": "" , "recommendations": ""}
            pest_folder_name = pest_folder_mapping[pest]
            base_path_meta = os.path.join(base_path , crop , "train" , "pests" , pest_folder_name)
            for key in metadata:
                possible_files = [f"{key}.txt" , f"{key}s.txt"] if key == "prevention" else [f"{key}.txt"]
                for file_name in possible_files:
                    file_path = os.path.join(base_path_meta , file_name)
                    if os.path.exists(file_path):
                        with open(file_path , "r" , encoding="utf-8") as f:
                            metadata[key] = f.read().strip()
                        break
            metadata_dict[pest] = metadata

        for health in healthy[crop]:
            metadata_dict[health] = {"causes": "" , "prevention": "" , "recommendations": ""}


# Custom Dataset class
class AgroSenseDataset(Dataset):
    def __init__(self , image_paths , labels , transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found: {path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self , idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            img = io.read_image(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None , None , img_path
        if self.transform:
            img = self.transform(img)
        return img , torch.tensor(label , dtype=torch.long) , img_path


# Data loading for test
def load_test_data():
    image_paths = []
    labels = []
    for crop in crops:
        healthy_base_path = os.path.join(base_path , crop , "test" , "crops")
        if os.path.exists(healthy_base_path):
            healthy_label = class_mapping[f"{crop.capitalize()}___healthy"]
            for filename in os.listdir(healthy_base_path):
                if filename.lower().endswith(('.png' , '.jpg' , '.jpeg')):
                    image_paths.append(os.path.join(healthy_base_path , filename))
                    labels.append(healthy_label)
            print(
                f"Loaded {len([f for f in os.listdir(healthy_base_path) if f.lower().endswith(('.png' , '.jpg' , '.jpeg'))])} healthy images from {healthy_base_path}")
        pests_base_path = os.path.join(base_path , crop , "test" , "pests")
        if os.path.exists(pests_base_path):
            for pest in pests[crop]:
                pest_folder_name = pest_folder_mapping[pest]
                pest_path = os.path.join(pests_base_path , pest_folder_name)
                if os.path.exists(pest_path):
                    for filename in os.listdir(pest_path):
                        if filename.lower().endswith(('.png' , '.jpg' , '.jpeg')):
                            image_paths.append(os.path.join(pest_path , filename))
                            labels.append(class_mapping[pest])
                    print(
                        f"Loaded {len([f for f in os.listdir(pest_path) if f.lower().endswith(('.png' , '.jpg' , '.jpeg'))])} images from {pest_path}")
                else:
                    print(f"Warning: Directory {pest_path} not found for {pest}")
    if not any(label in [0 , 8 , 14] for label in labels):
        raise ValueError("No healthy images found in the test dataset.")
    return image_paths , labels


# Format farmer-friendly output
def format_farm_output(class_name , crop , metadata):
    metadata = {
        "causes": metadata.get("causes" , "") ,
        "prevention": metadata.get("prevention" , "") ,
        "recommendations": metadata.get("recommendations" , "")
    }
    if class_name in healthy[crop]:
        return f"Your {crop} crop looks healthy! Keep up the good work with your current practices."
    output = f"**Diagnosis for Your {crop.capitalize()} Crop**\n"
    if class_name in diseases[crop]:
        output += f"Your {crop} crop is affected by a disease: **{class_name.replace(f'{crop.capitalize()}___' , '')}**.\n"
    else:
        display_name = class_name.replace(f"{crop.capitalize()}_" , "")
        output += f"Your {crop} crop is affected by a pest: **{display_name}**.\n"
    output += f"\n**What Causes This?**\n{metadata['causes'] if metadata['causes'] else 'No information available'}\n"
    output += f"\n**How to Prevent It?**\n{metadata['prevention'] if metadata['prevention'] else 'No information available'}\n"
    output += f"\n**What You Can Do Now:**\n{metadata['recommendations'] if metadata['recommendations'] else 'No information available'}\n"
    return output


# Data transformations for test
val_test_transform = transforms.Compose([
    transforms.Resize((224 , 224)) ,
    transforms.ConvertImageDtype(torch.float) ,
    transforms.Normalize(mean=[0.485 , 0.456 , 0.406] , std=[0.229 , 0.224 , 0.225])
])

if __name__ == '__main__':
    # Pre-load metadata
    print("Pre-loading metadata for all classes...")
    preload_metadata()

    # Load test data
    test_paths , test_labels = load_test_data()
    test_dataset = AgroSenseDataset(test_paths , test_labels , transform=val_test_transform)
    test_loader = DataLoader(test_dataset , batch_size=8 , shuffle=False , num_workers=2 , pin_memory=False)

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5) ,
        nn.Linear(num_ftrs , 256) ,
        nn.ReLU() ,
        nn.Dropout(0.3) ,
        nn.Linear(256 , len(class_names))  # 27 classes
    )
    model_path = r"/app/multi_class_classifier2.pth"
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Test evaluation with detailed performance tracking
    all_preds = []
    all_labels = []
    all_paths = []
    correct_count = 0
    incorrect_count = 0
    unpredicted_count = 0
    failed_images = []

    with torch.no_grad():
        for images , labels , paths in tqdm(test_loader , desc="Test Evaluation"):
            if images is None or labels is None:
                unpredicted_count += len(paths)
                failed_images.extend(paths)
                continue
            images , labels = images.to(device) , labels.to(device)
            try:
                outputs = model(images)
                _ , predicted = torch.max(outputs.data , 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_paths.extend(paths)
                correct_count += torch.sum(predicted == labels).item()
                incorrect_count += torch.sum(predicted != labels).item()
            except Exception as e:
                print(f"Error during prediction for batch at {paths[0]}: {e}")
                unpredicted_count += len(paths)
                failed_images.extend(paths)

    total_images = len(test_paths)
    accuracy = 100 * correct_count / (total_images - unpredicted_count) if (total_images - unpredicted_count) > 0 else 0
    conf_matrix = confusion_matrix(all_labels , all_preds) if all_preds and all_labels else np.array([])
    precision , recall , f1 , _ = precision_recall_fscore_support(all_labels , all_preds , average='weighted' ,
                                                                  zero_division=0) if all_preds and all_labels else (
    0 , 0 , 0 , None)

    print(f"Total Images: {total_images}")
    print(f"Correct Predictions: {correct_count}")
    print(f"Incorrect Predictions: {incorrect_count}")
    print(f"Unpredicted Instances: {unpredicted_count} (Failed images: {failed_images})")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix if conf_matrix.size else "No valid predictions to compute matrix")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    # Detailed predictions with metadata
    print("\nDetailed Predictions:")
    for path , pred , label in zip(all_paths , all_preds , all_labels):
        pred_class = class_names[pred]
        true_class = class_names[label]
        crop = "apple" if "apple" in path.lower() else "potato" if "potato" in path.lower() else "tomato"
        metadata = metadata_dict.get(pred_class , {"causes": "" , "prevention": "" , "recommendations": ""})
        farm_output = format_farm_output(pred_class , crop , metadata)
        print(f"\nImage: {path}")
        print(f"Predicted: {pred_class}, True: {true_class}")
        print(farm_output)