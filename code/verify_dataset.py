import os

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

# Count images and other files
total_images = 0
for crop in crops:
    print(f"\nCrop: {crop}")
    for split in splits:
        print(f"  Split: {split}")
        if split in ["train", "valid"]:
            for category in ["healthy", "diseased", "pests"]:
                if category == "healthy":
                    base_category_path = os.path.join(base_path, crop, split, "crops", category)
                    if os.path.exists(base_category_path):
                        for health in healthy[crop]:
                            health_path = os.path.join(base_category_path, health)
                            if os.path.exists(health_path):
                                image_count = len([f for f in os.listdir(health_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                                total_images += image_count
                                print(f"    {category}/{health}: {image_count} images")
                elif category == "diseased":
                    base_category_path = os.path.join(base_path, crop, split, "crops", category)
                    if os.path.exists(base_category_path):
                        for disease in diseases[crop]:
                            disease_path = os.path.join(base_category_path, disease)
                            if os.path.exists(disease_path):
                                images_path = os.path.join(disease_path, "images")
                                image_count = 0
                                if os.path.exists(images_path):
                                    image_count = len([f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                                    total_images += image_count
                                other_files = len([f for f in os.listdir(disease_path) if os.path.isfile(os.path.join(disease_path, f)) and not f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                                print(f"    {category}/{disease}: {image_count} images, {other_files} other files")
                elif category == "pests":
                    base_category_path = os.path.join(base_path, crop, split, "pests")
                    if os.path.exists(base_category_path):
                        for pest in pests[crop]:
                            pest_path = os.path.join(base_category_path, pest)
                            if os.path.exists(pest_path):
                                images_path = os.path.join(pest_path, "images")
                                image_count = 0
                                if os.path.exists(images_path):
                                    image_count = len([f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                                    total_images += image_count
                                other_files = len([f for f in os.listdir(pest_path) if os.path.isfile(os.path.join(pest_path, f)) and not f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                                print(f"    {category}/{pest}: {image_count} images, {other_files} other files")
        elif split == "test":
            base_category_path_crops = os.path.join(base_path, crop, split, "crops")
            if os.path.exists(base_category_path_crops):
                print(f"  Checking: {base_category_path_crops}")
                image_count = len([f for f in os.listdir(base_category_path_crops) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                total_images += image_count
                other_files = len([f for f in os.listdir(base_category_path_crops) if os.path.isfile(os.path.join(base_category_path_crops, f)) and not f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"    crops: {image_count} images, {other_files} other files")
            base_category_path_pests = os.path.join(base_path, crop, split, "pests")
            if os.path.exists(base_category_path_pests):
                print(f"  Checking: {base_category_path_pests}")
                for pest in pests[crop]:
                    pest_path = os.path.join(base_category_path_pests, pest)
                    if os.path.exists(pest_path):
                        image_count = len([f for f in os.listdir(pest_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                        total_images += image_count
                        other_files = len([f for f in os.listdir(pest_path) if os.path.isfile(os.path.join(pest_path, f)) and not f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                        print(f"    pests/{pest}: {image_count} images, {other_files} other files")

print(f"\nTotal image count: {total_images}")