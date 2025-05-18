import os
import cv2
import numpy as np
from pathlib import Path

# Dataset paths
base_path = r"C:\Users\91905\PycharmProjects\AgroSense_AI_v3\datasets\AgroSense"
crops = ["apple", "potato", "tomato"]
splits = ["train", "valid", "test"]
diseases = {
    "apple": ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust"],
    "potato": ["Potato___Early_blight", "Potato___Late_blight"],
    "tomato": ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
               "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Target_Spot",
               "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"]
}
pests = {
    "apple": ["Anthracnose", "Aphids", "Fruit_Fly", "Powdery_Mildew"],
    "potato": ["Potato_Beetle", "Tomato_Late_Blight", "Potato_Early_blight"],
    "tomato": ["Powdery_Mildew", "Aphids", "Tomato_Leaf_Curl_Virus", "Tomato_Late_blight"]
}
healthy = {
    "apple": ["Apple___healthy"],
    "potato": ["Potato___healthy"],
    "tomato": ["Tomato___healthy"]
}

# Preprocessing parameters
target_size = (224, 224)  # Standard size for models like ResNet
output_dir = r"C:\Users\91905\PycharmProjects\AgroSense_AI_v3\processed_datasets"

def preprocess_image(image_path, target_size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize to [0, 1]
    return img

def process_dataset():
    for crop in crops:
        for split in splits:
            # Healthy
            if split in ["train", "valid"]:
                healthy_base_path = os.path.join(base_path, crop, split, "crops", "healthy")
                print(f"Processing healthy path: {healthy_base_path}")
                for health in healthy[crop]:
                    health_path = os.path.join(healthy_base_path, health)
                    if os.path.exists(health_path):
                        output_health_path = os.path.join(output_dir, crop, split, "crops", "healthy", health)
                        Path(output_health_path).mkdir(parents=True, exist_ok=True)
                        for filename in os.listdir(health_path):
                            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                                img_path = os.path.join(health_path, filename)
                                processed_img = preprocess_image(img_path, target_size)
                                output_path = os.path.join(output_health_path, filename)
                                cv2.imwrite(output_path, (processed_img * 255).astype(np.uint8))
                    else:
                        print(f"Warning: {health_path} not found!")

                # Diseased
                diseased_base_path = os.path.join(base_path, crop, split, "crops", "diseased")
                print(f"Processing diseased path: {diseased_base_path}")
                for disease in diseases[crop]:
                    disease_path = os.path.join(diseased_base_path, disease)
                    if os.path.exists(disease_path):
                        images_path = os.path.join(disease_path, "images")
                        if os.path.exists(images_path):
                            output_disease_path = os.path.join(output_dir, crop, split, "crops", "diseased", disease, "images")
                            Path(output_disease_path).mkdir(parents=True, exist_ok=True)
                            for filename in os.listdir(images_path):
                                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    img_path = os.path.join(images_path, filename)
                                    processed_img = preprocess_image(img_path, target_size)
                                    output_path = os.path.join(output_disease_path, filename)
                                    cv2.imwrite(output_path, (processed_img * 255).astype(np.uint8))
                        # Copy .txt files
                        for txt_file in ["causes.txt", "preventions.txt", "requirements.txt"]:
                            txt_path = os.path.join(disease_path, txt_file)
                            if os.path.exists(txt_path):
                                output_txt_dir = os.path.join(output_dir, crop, split, "crops", "diseased", disease)
                                Path(output_txt_dir).mkdir(parents=True, exist_ok=True)
                                output_txt_path = os.path.join(output_txt_dir, txt_file)
                                with open(txt_path, 'r') as src, open(output_txt_path, 'w') as dst:
                                    dst.write(src.read())
                    else:
                        print(f"Warning: {disease_path} not found!")

                # Pests
                pests_base_path = os.path.join(base_path, crop, split, "pests")
                print(f"Processing pests path: {pests_base_path}")
                for pest in pests[crop]:
                    pest_path = os.path.join(pests_base_path, pest)
                    if os.path.exists(pest_path):
                        images_path = os.path.join(pest_path, "images")
                        if os.path.exists(images_path):
                            output_pest_path = os.path.join(output_dir, crop, split, "pests", pest, "images")
                            Path(output_pest_path).mkdir(parents=True, exist_ok=True)
                            for filename in os.listdir(images_path):
                                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    img_path = os.path.join(images_path, filename)
                                    processed_img = preprocess_image(img_path, target_size)
                                    output_path = os.path.join(output_pest_path, filename)
                                    cv2.imwrite(output_path, (processed_img * 255).astype(np.uint8))
                        # Copy .txt files
                        for txt_file in ["causes.txt", "preventions.txt", "requirements.txt"]:
                            txt_path = os.path.join(pest_path, txt_file)
                            if os.path.exists(txt_path):
                                output_txt_dir = os.path.join(output_dir, crop, split, "pests", pest)
                                Path(output_txt_dir).mkdir(parents=True, exist_ok=True)
                                output_txt_path = os.path.join(output_txt_dir, txt_file)
                                with open(txt_path, 'r') as src, open(output_txt_path, 'w') as dst:
                                    dst.write(src.read())
                    else:
                        print(f"Warning: {pest_path} not found!")

            # Test split
            elif split == "test":
                crops_path = os.path.join(base_path, crop, split, "crops")
                if os.path.exists(crops_path):
                    output_crops_path = os.path.join(output_dir, crop, split, "crops")
                    Path(output_crops_path).mkdir(parents=True, exist_ok=True)
                    for filename in os.listdir(crops_path):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(crops_path, filename)
                            processed_img = preprocess_image(img_path, target_size)
                            output_path = os.path.join(output_crops_path, filename)
                            cv2.imwrite(output_path, (processed_img * 255).astype(np.uint8))

                pests_path = os.path.join(base_path, crop, split, "pests")
                if os.path.exists(pests_path):
                    for pest in pests[crop]:
                        pest_path = os.path.join(pests_path, pest)
                        if os.path.exists(pest_path):
                            output_pest_path = os.path.join(output_dir, crop, split, "pests", pest)
                            Path(output_pest_path).mkdir(parents=True, exist_ok=True)
                            for filename in os.listdir(pest_path):
                                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    img_path = os.path.join(pest_path, filename)
                                    processed_img = preprocess_image(img_path, target_size)
                                    output_path = os.path.join(output_pest_path, filename)
                                    cv2.imwrite(output_path, (processed_img * 255).astype(np.uint8))

if __name__ == "__main__":
    process_dataset()
    print("Preprocessing completed. Processed images saved to:", output_dir)