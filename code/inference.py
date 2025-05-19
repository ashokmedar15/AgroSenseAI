import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models , transforms
from PIL import Image
import time

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
                        print(f"Loaded metadata file {file_path} for class {disease}")
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
                        print(f"Loaded metadata file {file_path} for class {pest}")
                        break
            metadata_dict[pest] = metadata

        for health in healthy[crop]:
            metadata_dict[health] = {"causes": "" , "prevention": "" , "recommendations": ""}
            print(f"Initialized empty metadata for healthy class {health}")


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


# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5) ,
    nn.Linear(num_ftrs , 256) ,
    nn.ReLU() ,
    nn.Dropout(0.3) ,
    nn.Linear(256 , len(class_names))
)
model_path = r"/app/multi_class_classifier2.pth"
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# Preload metadata at startup
print("Pre-loading metadata for all classes...")
preload_metadata()

# Define transformation for input image
val_test_transform = transforms.Compose([
    transforms.Resize((224 , 224)) ,
    transforms.ToTensor() ,
    transforms.Normalize(mean=[0.485 , 0.456 , 0.406] , std=[0.229 , 0.224 , 0.225])
])


def predict_image(image_path):
    """Perform inference on a single image and return the prediction with details."""
    start_time = time.time()
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = val_test_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            _ , predicted = torch.max(output , 1)
            pred_class = class_names[predicted.item()]
            crop = "apple" if "apple" in image_path.lower() else "potato" if "potato" in image_path.lower() else "tomato"
            metadata = metadata_dict.get(pred_class , {"causes": "" , "prevention": "" , "recommendations": ""})
            farm_output = format_farm_output(pred_class , crop , metadata)
        latency = (time.time() - start_time) * 1000  # Latency in milliseconds
        return pred_class , farm_output , latency , None
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        return None , None , latency , f"Error loading or processing image: {str(e)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for AgroSense AI model.")
    parser.add_argument('--image' , type=str , help="Path to a single image for inference.")
    parser.add_argument('--directory' , type=str , help="Path to a directory of images for batch inference.")
    args = parser.parse_args()

    if not args.image and not args.directory:
        print(
            "Error: No arguments provided. Please provide either an image path (--image) or a directory path (--directory).")
        print("Alternatively, enter an image path manually when prompted (leave blank to exit).")
        image_path = input("Enter image path: ").strip().strip('"')  # Remove quotes from input
        if image_path:
            args.image = image_path
        else:
            print("Exiting...")
            exit(0)

    if args.image and args.directory:
        print("Error: Please provide either an image path or a directory path, not both.")
        exit(1)

    # Single image inference
    if args.image:
        print(f"\nProcessing image: {args.image}")
        pred_class , farm_output , latency , error = predict_image(args.image)
        if error:
            print(f"Error: {error}")
        else:
            print(f"Prediction: {pred_class}")
            print(f"Details:\n{farm_output}")
            print(f"Latency: {latency:.2f}ms")

    # Batch inference from directory
    if args.directory:
        if not os.path.isdir(args.directory):
            print(f"Error: Directory {args.directory} does not exist.")
            exit(1)

        image_extensions = ('.png' , '.jpg' , '.jpeg')
        image_paths = [os.path.join(args.directory , f) for f in os.listdir(args.directory) if
                       f.lower().endswith(image_extensions)]
        if not image_paths:
            print(f"No images found in directory {args.directory}.")
            exit(1)

        total_images = len(image_paths)
        successful_predictions = 0
        total_latency = 0
        failed_images = []

        print(f"\nProcessing {total_images} images in directory: {args.directory}")
        for image_path in image_paths:
            print(f"\nProcessing image: {image_path}")
            pred_class , farm_output , latency , error = predict_image(image_path)
            total_latency += latency
            if error:
                print(f"Error: {error}")
                failed_images.append(image_path)
            else:
                print(f"Prediction: {pred_class}")
                print(f"Details:\n{farm_output}")
                print(f"Latency: {latency:.2f}ms")
                successful_predictions += 1

        print("\nSummary:")
        print(f"Total Images: {total_images}")
        print(f"Successful Predictions: {successful_predictions}")
        print(f"Failed Predictions: {total_images - successful_predictions}")
        print(f"Average Latency: {total_latency / total_images:.2f}ms")
        if failed_images:
            print(f"Failed Images: {failed_images}")