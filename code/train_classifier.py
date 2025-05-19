import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset , DataLoader
from torchvision import models , transforms , io
from sklearn.metrics import confusion_matrix , precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import random
import time
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Dataset paths
base_path = r"C:\Users\91905\PycharmProjects\AgroSense_AI_v3\datasets\AgroSense"
crops = ["apple" , "potato" , "tomato"]
splits = ["train" , "valid" , "test"]
diseases = {
    "apple": ["Apple___Apple_scab" , "Apple___Black_rot" , "Apple___Cedar_apple_rust"] ,
    "potato": ["Potato___Early_blight" , "Potato___Late_blight"] ,
    "tomato": ["Tomato___Bacterial_spot" , "Tomato___Early_blight" , "Tomato___Late_blight" ,
               "Tomato___Leaf_Mold" , "Tomato___Septoria_leaf_spot" , "Tomato___Target_Spot" ,
               "Tomato___Tomato_mosaic_virus" , "Tomato___Tomato_Yellow_Leaf_Curl_Virus"]
}
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

# Pest folder mapping (actual folder names in the dataset)
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

# Define class mapping (27 unique classes)
class_mapping = {
    "Apple___healthy": 0 ,
    "Apple___Apple_scab": 1 ,
    "Apple___Black_rot": 2 ,
    "Apple___Cedar_apple_rust": 3 ,
    "Apple_Anthracnose": 4 ,  # Apple pest
    "Apple_Aphids": 5 ,  # Apple pest
    "Apple_Fruit_Fly": 6 ,  # Apple pest
    "Apple_Powdery_Mildew": 7 ,  # Apple pest
    "Potato___healthy": 8 ,
    "Potato___Early_blight": 9 ,
    "Potato___Late_blight": 10 ,
    "Potato_Beetle": 11 ,  # Potato pest
    "Potato_Tomato_Late_Blight": 12 ,  # Potato pest
    "Potato_Early_blight": 13 ,  # Potato pest
    "Tomato___healthy": 14 ,
    "Tomato___Bacterial_spot": 15 ,
    "Tomato___Early_blight": 16 ,
    "Tomato___Late_blight": 17 ,
    "Tomato___Leaf_Mold": 18 ,
    "Tomato___Septoria_leaf_spot": 19 ,
    "Tomato___Target_Spot": 20 ,
    "Tomato___Tomato_mosaic_virus": 21 ,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 22 ,
    "Tomato_Powdery_Mildew": 23 ,  # Tomato pest
    "Tomato_Aphids": 24 ,  # Tomato pest
    "Tomato_Leaf_Curl_Virus": 25 ,  # Tomato pest
    "Tomato_Late_blight": 26  # Tomato pest
}
class_names = list(class_mapping.keys())

# Pre-load metadata for all classes
metadata_dict = {}


def preload_metadata():
    for crop in crops:
        # Process diseases
        for disease in diseases[crop]:
            metadata = {"causes": "" , "preventions": "" , "recommendations": ""}
            base_path_meta = os.path.join(base_path , crop , "train" , "crops" , "diseased" , disease)
            for key in metadata:
                file_path = os.path.join(base_path_meta , f"{key}.txt")
                if os.path.exists(file_path):
                    with open(file_path , "r" , encoding="utf-8") as f:
                        metadata[key] = f.read().strip()
                else:
                    print(f"Warning: Metadata file {file_path} not found for class {disease}")
            metadata_dict[disease] = metadata

        # Process pests
        for pest in pests[crop]:
            metadata = {"causes": "" , "preventions": "" , "recommendations": ""}
            pest_folder_name = pest_folder_mapping[pest]
            base_path_meta = os.path.join(base_path , crop , "train" , "pests" , pest_folder_name)
            for key in metadata:
                file_path = os.path.join(base_path_meta , f"{key}.txt")
                if os.path.exists(file_path):
                    with open(file_path , "r" , encoding="utf-8") as f:
                        metadata[key] = f.read().strip()
                else:
                    print(f"Warning: Metadata file {file_path} not found for class {pest}")
            metadata_dict[pest] = metadata

        # Healthy classes (no metadata files expected)
        for health in healthy[crop]:
            metadata_dict[health] = {"causes": "" , "preventions": "" , "recommendations": ""}


# Count images and other files
def count_images():
    total_images = 0
    for crop in crops:
        print(f"\nCrop: {crop}")
        for split in splits:
            print(f"  Split: {split}")
            if split in ["train" , "valid"]:
                for category in ["healthy" , "diseased" , "pests"]:
                    if category == "healthy":
                        base_category_path = os.path.join(base_path , crop , split , "crops" , category)
                        if os.path.exists(base_category_path):
                            for health in healthy[crop]:
                                health_path = os.path.join(base_category_path , health)
                                if os.path.exists(health_path):
                                    image_count = len([f for f in os.listdir(health_path) if
                                                       f.lower().endswith(('.png' , '.jpg' , '.jpeg'))])
                                    total_images += image_count
                                    print(f"    {category}/{health}: {image_count} images")
                    elif category == "diseased":
                        base_category_path = os.path.join(base_path , crop , split , "crops" , "diseased")
                        if os.path.exists(base_category_path):
                            for disease in diseases[crop]:
                                disease_path = os.path.join(base_category_path , disease)
                                if os.path.exists(disease_path):
                                    images_path = os.path.join(disease_path , "images")
                                    image_count = 0
                                    if os.path.exists(images_path):
                                        image_count = len([f for f in os.listdir(images_path) if
                                                           f.lower().endswith(('.png' , '.jpg' , '.jpeg'))])
                                        total_images += image_count
                                    other_files = len([f for f in os.listdir(disease_path) if os.path.isfile(
                                        os.path.join(disease_path , f)) and not f.lower().endswith(
                                        ('.png' , '.jpg' , '.jpeg'))])
                                    print(f"    {category}/{disease}: {image_count} images, {other_files} other files")
                    elif category == "pests":
                        base_category_path = os.path.join(base_path , crop , split , "pests")
                        if os.path.exists(base_category_path):
                            for pest in pests[crop]:
                                pest_folder_name = pest_folder_mapping[pest]
                                pest_path = os.path.join(base_category_path , pest_folder_name)
                                if os.path.exists(pest_path):
                                    images_path = os.path.join(pest_path , "images")
                                    image_count = 0
                                    if os.path.exists(images_path):
                                        image_count = len([f for f in os.listdir(images_path) if
                                                           f.lower().endswith(('.png' , '.jpg' , '.jpeg'))])
                                        total_images += image_count
                                    other_files = len([f for f in os.listdir(pest_path) if os.path.isfile(
                                        os.path.join(pest_path , f)) and not f.lower().endswith(
                                        ('.png' , '.jpg' , '.jpeg'))])
                                    print(
                                        f"    {category}/{pest_folder_name}: {image_count} images, {other_files} other files")
                                else:
                                    print(f"    Warning: Directory {pest_path} not found for {pest}")
            elif split == "test":
                for category in ["crops" , "pests"]:
                    if category == "crops":
                        base_category_path = os.path.join(base_path , crop , split , category)
                        if os.path.exists(base_category_path):
                            print(f"  Checking: {base_category_path}")
                            image_count = len([f for f in os.listdir(base_category_path) if
                                               f.lower().endswith(('.png' , '.jpg' , '.jpeg'))])
                            total_images += image_count
                            print(f"    {category}: {image_count} images")
                    elif category == "pests":
                        base_category_path = os.path.join(base_path , crop , split , "pests")
                        if os.path.exists(base_category_path):
                            print(f"  Checking: {base_category_path}")
                            for pest in pests[crop]:
                                pest_folder_name = pest_folder_mapping[pest]
                                pest_path = os.path.join(base_category_path , pest_folder_name)
                                if os.path.exists(pest_path):
                                    image_count = len([f for f in os.listdir(pest_path) if
                                                       f.lower().endswith(('.png' , '.jpg' , '.jpeg'))])
                                    total_images += image_count
                                    print(f"    {category}/{pest_folder_name}: {image_count} images")
                                else:
                                    print(f"    Warning: Directory {pest_path} not found for {pest}")
    print(f"\nTotal image count: {total_images}")
    return total_images


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
            img = torch.zeros((3 , 224 , 224))
            label = 0
        if self.transform:
            img = self.transform(img)
        return img , torch.tensor(label , dtype=torch.long) , img_path


# Data loading for training
def load_train_data():
    image_paths = []
    labels = []
    for crop in crops:
        for split in ["train" , "valid"]:
            # Healthy
            healthy_base_path = os.path.join(base_path , crop , split , "crops" , "healthy")
            if os.path.exists(healthy_base_path):
                for health in healthy[crop]:
                    health_path = os.path.join(healthy_base_path , health)
                    if os.path.exists(health_path):
                        for filename in os.listdir(health_path):
                            if filename.lower().endswith(('.png' , '.jpg' , '.jpeg')):
                                image_paths.append(os.path.join(health_path , filename))
                                labels.append(class_mapping[health])
                        print(
                            f"Loaded {len([f for f in os.listdir(health_path) if f.lower().endswith(('.png' , '.jpg' , '.jpeg'))])} images from {health_path}")
            # Diseased
            diseased_base_path = os.path.join(base_path , crop , split , "crops" , "diseased")
            if os.path.exists(diseased_base_path):
                for disease in diseases[crop]:
                    disease_path = os.path.join(diseased_base_path , disease , "images")
                    if os.path.exists(disease_path):
                        for filename in os.listdir(disease_path):
                            if filename.lower().endswith(('.png' , '.jpg' , '.jpeg')):
                                image_paths.append(os.path.join(disease_path , filename))
                                labels.append(class_mapping[disease])
                        print(
                            f"Loaded {len([f for f in os.listdir(disease_path) if f.lower().endswith(('.png' , '.jpg' , '.jpeg'))])} images from {disease_path}")
            # Pests
            pests_base_path = os.path.join(base_path , crop , split , "pests")
            if os.path.exists(pests_base_path):
                for pest in pests[crop]:
                    pest_folder_name = pest_folder_mapping[pest]
                    pest_path = os.path.join(pests_base_path , pest_folder_name , "images")
                    if os.path.exists(pest_path):
                        for filename in os.listdir(pest_path):
                            if filename.lower().endswith(('.png' , '.jpg' , '.jpeg')):
                                image_paths.append(os.path.join(pest_path , filename))
                                labels.append(class_mapping[pest])
                        print(
                            f"Loaded {len([f for f in os.listdir(pest_path) if f.lower().endswith(('.png' , '.jpg' , '.jpeg'))])} images from {pest_path}")
                    else:
                        print(f"Warning: Directory {pest_path} not found for {pest}")
    print(f"Total training images loaded: {len(image_paths)} with {len(set(labels))} unique classes")
    return image_paths , labels


# Data loading for test
def load_test_data():
    image_paths = []
    labels = []
    for crop in crops:
        # Healthy (crops category for test)
        healthy_base_path = os.path.join(base_path , crop , "test" , "crops")
        if os.path.exists(healthy_base_path):
            healthy_label = class_mapping[f"{crop.capitalize()}___healthy"]
            for filename in os.listdir(healthy_base_path):
                if filename.lower().endswith(('.png' , '.jpg' , '.jpeg')):
                    image_paths.append(os.path.join(healthy_base_path , filename))
                    labels.append(healthy_label)
            print(
                f"Loaded {len([f for f in os.listdir(healthy_base_path) if f.lower().endswith(('.png' , '.jpg' , '.jpeg'))])} healthy images from {healthy_base_path}")
        # Unhealthy (pests category for test)
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
    if class_name in healthy[crop]:
        return f"Your {crop} crop looks healthy! Keep up the good work with your current practices."

    output = f"**Diagnosis for Your {crop.capitalize()} Crop**\n"
    if class_name in diseases[crop]:
        output += f"Your {crop} crop is affected by a disease: **{class_name.replace(f'{crop.capitalize()}___' , '')}**.\n"
    else:
        # Remove the crop prefix from pest name for display
        display_name = class_name.replace(f"{crop.capitalize()}_" , "")
        output += f"Your {crop} crop is affected by a pest: **{display_name}**.\n"

    if metadata["causes"]:
        output += f"\n**What Causes This?**\n{metadata['causes']}\n"
    else:
        output += f"\n**What Causes This?**\n(No information available)\n"
    if metadata["prevention"]:
        output += f"\n**How to Prevent It?**\n{metadata['prevention']}\n"
    else:
        output += f"\n**How to Prevent It?**\n(No information available)\n"
    if metadata["recommendations"]:
        output += f"\n**What You Can Do Now:**\n{metadata['recommendations']}\n"
    else:
        output += f"\n**What You Can Do Now:**\n(No information available)\n"

    return output


# Data transformations
train_transform = transforms.Compose([
    transforms.Resize((224 , 224)) ,
    transforms.RandomHorizontalFlip() ,
    transforms.RandomVerticalFlip() ,
    transforms.RandomRotation(15) ,
    transforms.ColorJitter(brightness=0.2 , contrast=0.2 , saturation=0.2) ,
    transforms.ConvertImageDtype(torch.float) ,
    transforms.Normalize(mean=[0.485 , 0.456 , 0.406] , std=[0.229 , 0.224 , 0.225])
])
val_test_transform = transforms.Compose([
    transforms.Resize((224 , 224)) ,
    transforms.ConvertImageDtype(torch.float) ,
    transforms.Normalize(mean=[0.485 , 0.456 , 0.406] , std=[0.229 , 0.224 , 0.225])
])

if __name__ == '__main__':
    # Pre-load metadata for all classes
    print("Pre-loading metadata for all classes...")
    preload_metadata()

    # Count images across all splits
    print("Counting images in all splits...")
    total_images = count_images()

    # Load training data
    start_time = time.time()
    image_paths , labels = load_train_data()
    print(f"Training data loading completed. Time taken: {time.time() - start_time:.2f} seconds")

    # Compute class weights
    class_counts = np.bincount(labels , minlength=len(class_names))
    print(f"Class counts: {class_counts}")
    # Replace zero counts with 1 to avoid division by zero
    class_counts = np.where(class_counts == 0 , 1 , class_counts)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_names) * class_counts)
    class_weights = torch.FloatTensor(class_weights)
    print(f"Class weights: {class_weights}")

    # Split data into train and validation sets
    train_indices , val_indices = train_test_split(
        range(len(image_paths)) , test_size=0.2 , stratify=labels , random_state=42
    )

    # Create datasets
    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    train_dataset = AgroSenseDataset(train_paths , train_labels , transform=train_transform)
    val_dataset = AgroSenseDataset(val_paths , val_labels , transform=val_test_transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset , batch_size=8 , shuffle=True , num_workers=2 , pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset , batch_size=8 , shuffle=False , num_workers=2 , pin_memory=False
    )

    # Model setup with progressive unfreezing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5) ,
        nn.Linear(num_ftrs , 256) ,
        nn.ReLU() ,
        nn.Dropout(0.3) ,
        nn.Linear(256 , len(class_names))  # 27 classes
    )
    model = model.to(device)

    # Loss and optimizer with scheduler
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad , model.parameters()) , lr=0.0001 , weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer , mode='min' , factor=0.1 , patience=3)

    # Training loop with progressive unfreezing
    num_epochs = 5
    best_val_acc = 0.0
    model_path = r"/app/multi_class_classifier2.pth"

    unfreeze_schedule = {2: model.layer3 , 4: model.layer2}

    for epoch in range(num_epochs):
        if epoch in unfreeze_schedule:
            for param in unfreeze_schedule[epoch].parameters():
                param.requires_grad = True
            optimizer = optim.Adam(filter(lambda p: p.requires_grad , model.parameters()) , lr=0.0001 ,
                                   weight_decay=1e-4)
            print(f"Unfrozen {unfreeze_schedule[epoch]} at epoch {epoch}")

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()

        for batch_idx , (images , labels , _) in enumerate(tqdm(train_loader , desc=f"Epoch {epoch + 1}/{num_epochs}")):
            images , labels = images.to(device) , labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs , labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _ , predicted = torch.max(outputs.data , 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%, Epoch Time: {time.time() - epoch_start:.2f} seconds")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images , labels , _ in tqdm(val_loader , desc="Validation"):
                images , labels = images.to(device) , labels.to(device)
                outputs = model(images)
                loss = criterion(outputs , labels)
                val_loss += loss.item()
                _ , predicted = torch.max(outputs.data , 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict() , model_path)
            print(f"Saved best model with validation accuracy: {best_val_acc:.2f}%")

    print("Training completed. Best model saved to:" , model_path)

    # Load test data
    test_paths , test_labels = load_test_data()
    test_dataset = AgroSenseDataset(test_paths , test_labels , transform=val_test_transform)
    test_loader = DataLoader(test_dataset , batch_size=8 , shuffle=False , num_workers=2 , pin_memory=False)

    # Load the trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Test evaluation
    all_preds = []
    all_labels = []
    all_paths = []
    with torch.no_grad():
        for images , labels , paths in tqdm(test_loader , desc="Test Evaluation"):
            images , labels = images.to(device) , labels.to(device)
            outputs = model(images)
            _ , predicted = torch.max(outputs.data , 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)

    # Compute metrics
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    conf_matrix = confusion_matrix(all_labels , all_preds)
    precision , recall , f1 , _ = precision_recall_fscore_support(all_labels , all_preds , average='weighted' ,
                                                                  zero_division=0)

    print(f"Test Accuracy: {accuracy:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    # Detailed predictions with metadata
    print("\nDetailed Predictions:")
    for path , pred , label in zip(all_paths , all_preds , all_labels):
        pred_class = class_names[pred]
        true_class = class_names[label]
        # Determine crop from path
        crop = "apple" if "apple" in path.lower() else "potato" if "potato" in path.lower() else "tomato"
        metadata = metadata_dict.get(pred_class , {"causes": "" , "preventions": "" , "recommendations": ""})
        farm_output = format_farm_output(pred_class , crop , metadata)
        print(f"\nImage: {path}")
        print(f"Predicted: {pred_class}, True: {true_class}")
        print(farm_output)