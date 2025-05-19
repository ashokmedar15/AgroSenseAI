import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models , transforms , io
from flask import Flask , request , render_template , jsonify
from PIL import Image
import io
import time
import logging
import base64
import traceback
from transformers import AutoModelForCausalLM , AutoTokenizer
from deep_translator import GoogleTranslator
import cv2
import threading
import functools


# Timeout decorator for cross-platform compatibility
class TimeoutError(Exception):
    pass


def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args , **kwargs):
            result = [TimeoutError("Function timed out")]

            def target():
                try:
                    result[0] = func(*args , **kwargs)
                except Exception as exception:
                    result[0] = exception

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)

            if isinstance(result[0] , Exception):
                raise result[0]
            return result[0]

        return wrapper

    return decorator


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

# Supported languages with their corresponding language codes for GoogleTranslator
SUPPORTED_LANGUAGES = {
    'en': 'en' ,  # English
    'hi': 'hi' ,  # Hindi
    'te': 'te' ,  # Telugu
    'ta': 'ta' ,  # Tamil
    'ml': 'ml' ,  # Malayalam
    'kn': 'kn'  # Kannada
}

# Default metadata for diseases, pests, and healthy crops
default_metadata = {
    # Apple Diseases
    "Apple___Apple_scab": {
        "causes": "Caused by the fungus Venturia inaequalis. It spreads through wet conditions in spring. Old leaves on the ground can carry the fungus and infect new leaves." ,
        "prevention": "Prune trees to improve air circulation. Apply fungicides during early spring. Clean up fallen leaves to stop the fungus from coming back." ,
        "recommendations": "Remove fallen leaves to reduce fungal spores. Monitor weather for wet conditions. Check your trees every few days for new spots."
    } ,
    "Apple___Black_rot": {
        "causes": "Caused by the fungus Botryosphaeria obtusa. It thrives in warm, humid weather. It can spread through infected fruit and branches." ,
        "prevention": "Sanitize pruning tools to avoid spreading. Apply fungicides during bloom. Ensure proper tree spacing for airflow." ,
        "recommendations": "Remove infected fruit and branches. Ensure proper tree spacing for airflow. Monitor for dark spots on fruit and leaves."
    } ,
    "Apple___Cedar_apple_rust": {
        "causes": "Caused by the fungus Gymnosporangium juniperi-virginianae. It needs cedar trees nearby to spread. It spreads faster in wet spring weather." ,
        "prevention": "Remove nearby cedar trees if possible. Apply fungicides in spring. Prune affected areas to stop the spread." ,
        "recommendations": "Monitor for orange spots on leaves. Prune affected areas immediately. Check nearby cedar trees for signs of the fungus."
    } ,
    # Apple Pests
    "Apple_Anthracnose": {
        "causes": "Caused by the fungus Colletotrichum gloeosporioides. It spreads in wet, warm conditions. Infected fruit can spread it to healthy ones." ,
        "prevention": "Avoid overhead watering to keep foliage dry. Apply fungicides during rainy seasons. Improve air circulation around trees." ,
        "recommendations": "Remove infected fruit and leaves. Improve air circulation around trees. Monitor during rainy seasons for new infections."
    } ,
    "Apple_Aphids": {
        "causes": "Caused by small sap-sucking insects. They thrive in warm, dry weather. They can spread viruses to your trees." ,
        "prevention": "Introduce beneficial insects like ladybugs. Spray with neem oil or insecticidal soap. Check leaves regularly for aphids." ,
        "recommendations": "Check leaves for sticky residue. Prune heavily infested branches. Use neem oil spray every week until aphids are gone."
    } ,
    "Apple_Fruit_Fly": {
        "causes": "Caused by fruit flies laying eggs in fruit. They are active in warm weather. Fallen fruit can attract more flies." ,
        "prevention": "Use traps to monitor and capture flies. Cover fruit with fine mesh bags. Clean up fallen fruit regularly." ,
        "recommendations": "Remove fallen fruit to break the life cycle. Apply organic insecticides if needed. Set up fruit fly traps around your orchard."
    } ,
    "Apple_Powdery_Mildew": {
        "causes": "Caused by the fungus Podosphaera leucotricha. It spreads in dry, warm conditions. It can weaken your trees over time." ,
        "prevention": "Prune trees to improve air circulation. Apply sulfur-based fungicides early. Avoid excessive nitrogen fertilizers." ,
        "recommendations": "Monitor for white powdery spots on leaves. Avoid excessive nitrogen fertilizers. Spray with sulfur fungicide if you see signs."
    } ,
    # Apple Healthy
    "Apple___healthy": {
        "causes": "No issues detected. Your trees are growing well. Keep up the good work." ,
        "prevention": "Maintain regular watering and fertilization. Monitor for early signs of pests or diseases. Prune trees to keep them healthy." ,
        "recommendations": "Continue good farming practices. Check trees weekly for any changes. Fertilize your trees as needed to keep them strong."
    } ,
    # Potato Diseases
    "Potato___Early_blight": {
        "causes": "Caused by the fungus Alternaria solani. It spreads in warm, wet conditions. It can live in soil and infect new plants." ,
        "prevention": "Rotate crops yearly to prevent buildup. Apply fungicides during humid weather. Avoid overhead watering to keep foliage dry." ,
        "recommendations": "Remove infected leaves carefully. Avoid overhead watering to keep foliage dry. Check plants regularly for dark spots."
    } ,
    "Potato___Late_blight": {
        "causes": "Caused by the oomycete Phytophthora infestans. It thrives in cool, wet weather. It can spread quickly through the air." ,
        "prevention": "Ensure proper spacing for air circulation. Apply fungicides before rainy seasons. Destroy infected plants to stop the spread." ,
        "recommendations": "Destroy infected plants immediately. Monitor weather forecasts for wet conditions. Use resistant potato varieties if possible."
    } ,
    # Potato Pests
    "Potato_Beetle": {
        "causes": "Caused by Colorado potato beetles. They feed on potato leaves. They can multiply quickly in warm weather." ,
        "prevention": "Hand-pick beetles and larvae daily. Use row covers to protect young plants. Rotate crops to disrupt their lifecycle." ,
        "recommendations": "Apply organic insecticides if infestation is heavy. Rotate crops to disrupt their lifecycle. Check plants daily for beetles."
    } ,
    "Potato_Tomato_Late_Blight": {
        "causes": "Caused by the oomycete Phytophthora infestans. It affects both potatoes and tomatoes. It spreads in cool, wet weather." ,
        "prevention": "Avoid planting potatoes and tomatoes together. Apply fungicides during wet weather. Improve soil drainage to reduce moisture." ,
        "recommendations": "Remove and destroy affected plants. Improve soil drainage to reduce moisture. Monitor nearby tomato plants for signs."
    } ,
    # Potato Healthy
    "Potato___healthy": {
        "causes": "No issues detected. Your potatoes are growing well. Keep up the good work." ,
        "prevention": "Maintain balanced soil nutrients. Monitor for early signs of pests or diseases. Rotate crops yearly to keep soil healthy." ,
        "recommendations": "Continue good farming practices. Check plants weekly for any changes. Add compost to soil to keep it rich."
    } ,
    # Tomato Diseases
    "Tomato___Bacterial_spot": {
        "causes": "Caused by Xanthomonas bacteria. It spreads through water splashes. It can live on plant debris." ,
        "prevention": "Avoid overhead watering to keep leaves dry. Use copper-based bactericides. Remove plant debris after harvest." ,
        "recommendations": "Remove infected leaves carefully. Ensure proper spacing for air circulation. Use copper spray during wet weather."
    } ,
    "Tomato___Early_blight": {
        "causes": "Caused by the fungus Alternaria solani. It thrives in warm, wet conditions. It can spread through soil and water." ,
        "prevention": "Stake plants to keep leaves off the ground. Apply fungicides during humid weather. Rotate crops yearly to prevent buildup." ,
        "recommendations": "Remove lower infected leaves. Rotate crops yearly to prevent buildup. Check plants after rain for spots."
    } ,
    "Tomato___Late_blight": {
        "causes": "Caused by the oomycete Phytophthora infestans. It spreads in cool, wet weather. It can spread through the air to nearby plants." ,
        "prevention": "Ensure good air circulation around plants. Apply fungicides before rainy seasons. Avoid watering late in the day." ,
        "recommendations": "Destroy infected plants immediately. Avoid watering late in the day. Use resistant tomato varieties if possible."
    } ,
    "Tomato___Leaf_Mold": {
        "causes": "Caused by the fungus Fulvia fulva. It thrives in high humidity. It spreads in greenhouses with poor ventilation." ,
        "prevention": "Increase ventilation in greenhouses. Avoid overhead watering. Prune plants to improve air flow." ,
        "recommendations": "Remove affected leaves carefully. Apply fungicides if humidity persists. Open greenhouse vents to reduce humidity."
    } ,
    "Tomato___Septoria_leaf_spot": {
        "causes": "Caused by the fungus Septoria lycopersici. It spreads through water splashes. It can live on plant debris." ,
        "prevention": "Stake plants to keep leaves off the ground. Apply fungicides during wet weather. Remove plant debris after harvest." ,
        "recommendations": "Remove infected leaves and debris. Improve air circulation around plants. Use fungicides after heavy rain."
    } ,
    "Tomato___Target_Spot": {
        "causes": "Caused by the fungus Corynespora cassiicola. It thrives in warm, humid conditions. It can spread through water splashes." ,
        "prevention": "Ensure proper spacing for air circulation. Apply fungicides during humid weather. Avoid overhead watering to keep foliage dry." ,
        "recommendations": "Remove affected leaves carefully. Avoid overhead watering to keep foliage dry. Check plants during humid weather."
    } ,
    "Tomato___Tomato_mosaic_virus": {
        "causes": "Caused by a virus spread through contact. It can persist in soil and tools. It can spread through handling infected plants." ,
        "prevention": "Sanitize tools and hands after handling plants. Avoid smoking near plants. Use resistant tomato varieties." ,
        "recommendations": "Remove and destroy infected plants. Use resistant tomato varieties. Wash hands after touching plants."
    } ,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "causes": "Caused by a virus spread by whiteflies. It thrives in warm weather. It can spread quickly in dense plantings." ,
        "prevention": "Use reflective mulches to repel whiteflies. Apply insecticides to control whiteflies. Space plants to reduce whitefly spread." ,
        "recommendations": "Remove infected plants immediately. Monitor for whitefly populations. Use sticky traps to catch whiteflies."
    } ,
    # Tomato Pests
    "Tomato_Powdery_Mildew": {
        "causes": "Caused by the fungus Oidium neolycopersici. It spreads in dry, warm conditions. It can weaken your plants over time." ,
        "prevention": "Prune plants to improve air circulation. Apply sulfur-based fungicides early. Avoid excessive nitrogen fertilizers." ,
        "recommendations": "Monitor for white powdery spots on leaves. Avoid excessive nitrogen fertilizers. Spray with sulfur fungicide if you see signs."
    } ,
    "Tomato_Aphids": {
        "causes": "Caused by small sap-sucking insects. They thrive in warm, dry weather. They can spread viruses to your plants." ,
        "prevention": "Introduce beneficial insects like ladybugs. Spray with neem oil or insecticidal soap. Check leaves regularly for aphids." ,
        "recommendations": "Check leaves for sticky residue. Prune heavily infested branches. Use neem oil spray every week until aphids are gone."
    } ,
    "Tomato_Leaf_Curl_Virus": {
        "causes": "Caused by a virus spread by whiteflies. It thrives in warm weather. It can spread quickly in dense plantings." ,
        "prevention": "Use reflective mulches to repel whiteflies. Apply insecticides to control whiteflies. Space plants to reduce whitefly spread." ,
        "recommendations": "Remove infected plants immediately. Monitor for whitefly populations. Use sticky traps to catch whiteflies."
    } ,
    # Tomato Healthy
    "Tomato___healthy": {
        "causes": "No issues detected. Your tomatoes are growing well. Keep up the good work." ,
        "prevention": "Maintain regular watering and fertilization. Monitor for early signs of pests or diseases. Stake plants to keep them healthy." ,
        "recommendations": "Continue good farming practices. Check plants weekly for any changes. Add compost to soil to keep it rich."
    }
}

# Damage factors for each class (0.0 to 1.0, where higher means more severe damage)
damage_factors = {
    # Healthy crops (low damage)
    "Apple___healthy": 0.1 ,
    "Potato___healthy": 0.1 ,
    "Tomato___healthy": 0.1 ,
    # Apple Diseases (moderate to high damage)
    "Apple___Apple_scab": 0.7 ,
    "Apple___Black_rot": 0.8 ,
    "Apple___Cedar_apple_rust": 0.6 ,
    # Apple Pests (moderate to high damage)
    "Apple_Anthracnose": 0.8 ,
    "Apple_Aphids": 0.5 ,
    "Apple_Fruit_Fly": 0.7 ,
    "Apple_Powdery_Mildew": 0.6 ,
    # Potato Diseases (moderate to high damage)
    "Potato___Early_blight": 0.6 ,
    "Potato___Late_blight": 0.9 ,
    # Potato Pests (moderate to high damage)
    "Potato_Beetle": 0.7 ,
    "Potato_Tomato_Late_Blight": 0.9 ,
    "Potato_Early_blight": 0.6 ,
    # Tomato Diseases (moderate to high damage)
    "Tomato___Bacterial_spot": 0.7 ,
    "Tomato___Early_blight": 0.6 ,
    "Tomato___Late_blight": 0.9 ,
    "Tomato___Leaf_Mold": 0.5 ,
    "Tomato___Septoria_leaf_spot": 0.6 ,
    "Tomato___Target_Spot": 0.6 ,
    "Tomato___Tomato_mosaic_virus": 0.8 ,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 0.9 ,
    # Tomato Pests (moderate to high damage)
    "Tomato_Powdery_Mildew": 0.6 ,
    "Tomato_Aphids": 0.5 ,
    "Tomato_Leaf_Curl_Virus": 0.9 ,
    "Tomato_Late_blight": 0.9
}

# Pre-load metadata
metadata_dict = {}


def preload_metadata():
    try:
        for crop in crops:
            for disease in diseases[crop]:
                metadata = {"causes": "" , "prevention": "" , "recommendations": ""}
                base_path_meta = os.path.join(base_path , crop , "train" , "crops" , "diseased" , disease)
                for key in metadata:
                    possible_files = [f"{key}.txt" , f"{key}s.txt"] if key == "prevention" else [f"{key}.txt"]
                    found = False
                    for file_name in possible_files:
                        file_path = os.path.join(base_path_meta , file_name)
                        if os.path.exists(file_path):
                            with open(file_path , "r" , encoding="utf-8") as f:
                                metadata[key] = f.read().strip()
                            found = True
                            break
                    if not found:
                        # Use default metadata if file not found
                        metadata[key] = default_metadata.get(disease , {}).get(key , "Not available")
                metadata_dict[disease] = metadata

            for pest in pests[crop]:
                metadata = {"causes": "" , "prevention": "" , "recommendations": ""}
                pest_folder_name = pest_folder_mapping[pest]
                base_path_meta = os.path.join(base_path , crop , "train" , "pests" , pest_folder_name)
                for key in metadata:
                    possible_files = [f"{key}.txt" , f"{key}s.txt"] if key == "prevention" else [f"{key}.txt"]
                    found = False
                    for file_name in possible_files:
                        file_path = os.path.join(base_path_meta , file_name)
                        if os.path.exists(file_path):
                            with open(file_path , "r" , encoding="utf-8") as f:
                                metadata[key] = f.read().strip()
                            found = True
                            break
                    if not found:
                        # Use default metadata if file not found
                        metadata[key] = default_metadata.get(pest , {}).get(key , "Not available")
                metadata_dict[pest] = metadata

            for health in healthy[crop]:
                metadata = {"causes": "" , "prevention": "" , "recommendations": ""}
                for key in metadata:
                    # Use default metadata for healthy crops
                    metadata[key] = default_metadata.get(health , {}).get(key , "Not available")
                metadata_dict[health] = metadata
        print("Metadata preloading completed successfully.")
    except Exception as e:
        print(f"Error during metadata preloading: {str(e)}")
        raise


# Determine the crop based on the predicted class
def determine_crop_from_class(pred_class):
    for crop in crops:
        if pred_class in healthy[crop]:
            return crop
        if pred_class in diseases[crop]:
            return crop
        if pred_class in pests[crop]:
            return crop
    return "unknown"  # Fallback if class not found (shouldn't happen)


# Function to batch translate texts using deep-translator
def batch_translate_texts(texts , target_lang):
    if target_lang == 'en':
        return texts  # Skip translation for English
    try:
        # Join all texts into a single string with a delimiter
        delimiter = " ||| "
        combined_text = delimiter.join(texts)
        translator = GoogleTranslator(source='en' , target=target_lang)
        translated_combined = translator.translate(combined_text)
        # Split the translated text back into a list
        translated_texts = translated_combined.split(delimiter)
        # Ensure the number of translated texts matches the input
        if len(translated_texts) != len(texts):
            logging.warning(f"Translation mismatch: Expected {len(texts)} texts, got {len(translated_texts)}")
            return texts  # Fallback to English if translation fails
        return translated_texts
    except Exception as e:
        logging.error(f"Batch translation error to {target_lang}: {str(e)}")
        return texts  # Fallback to English if translation fails


# Normalize duplicate class names (e.g., Potato_Early_blight)
def normalize_class_name(class_name):
    if class_name == "Potato_Early_blight":
        return "Potato___Early_blight"  # Map duplicate to the primary class name
    return class_name


# Format output for LLM processing with enhanced pest handling (raw English output)
def format_farm_output(class_name , crop , metadata , confidence , severity):
    class_name = normalize_class_name(class_name)
    if class_name in healthy[crop]:
        # For healthy crops, remove the crop prefix and underscores
        display_name = class_name.replace(f"{crop.capitalize()}___" , "").replace("_" , " ")
        message = f"Your {crop} crop is healthy."
        action = "Continue current practices."
        return {
            "status": "Healthy" ,
            "message": message ,
            "confidence": confidence ,
            "severity": severity ,
            "action": action ,
            "class_name": class_name ,
            "type": "healthy" ,
            "details": {
                "causes": metadata["causes"] ,
                "prevention": metadata["prevention"] ,
                "recommendations": metadata["recommendations"]
            }
        }
    issue_type = "pest" if class_name in pests[crop] else "disease"
    # For diseases and pests, remove the crop prefix and underscores
    display_name = class_name.replace(f"{crop.capitalize()}___" , "").replace(f"{crop.capitalize()}_" , "").replace("_" , " ")
    message = f"Your {crop} crop is affected by a {issue_type}: {display_name}."
    return {
        "status": "Affected" ,
        "message": message ,
        "confidence": confidence ,
        "severity": severity ,
        "details": {
            "causes": metadata["causes"] ,
            "prevention": metadata["prevention"] ,
            "recommendations": metadata["recommendations"]
        } ,
        "class_name": class_name ,
        "type": issue_type
    }


# Determine severity based on confidence with granular levels
def determine_severity(confidence):
    if confidence > 95:
        return "Very High"
    elif confidence > 80:
        return "High"
    elif confidence > 60:
        return "Moderate"
    elif confidence > 40:
        return "Mild"
    elif confidence > 20:
        return "Low"
    else:
        return "Very Low"


# Process base64 image data
def process_base64_image(base64_string):
    try:
        header , encoded = base64_string.split(',' , 1)
        img_data = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        print(f"Processed base64 image with size: {img.size}")
        return img
    except Exception as e:
        print(f"Error processing base64 image: {str(e)}")
        return None


# Process uploaded file
def process_uploaded_file(file):
    try:
        img = Image.open(file.stream).convert('RGB')
        print(f"Processed uploaded file with size: {img.size}")
        return img
    except Exception as e:
        print(f"Error processing uploaded file: {str(e)}")
        return None


# Calculate entropy of a probability distribution for OOD detection
def calculate_entropy(probabilities):
    return -np.sum(probabilities * np.log(probabilities + 1e-10))  # Add small epsilon to avoid log(0)


# Simple heuristic: Check color histogram for crop-like characteristics
def is_crop_like_image(img):
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(img)
        if len(img_array.shape) != 3:  # Ensure it's an RGB image
            return False

        # Calculate histogram for each channel (R, G, B)
        hist_r = np.histogram(img_array[: , : , 0] , bins=256 , range=(0 , 256))[0]
        hist_g = np.histogram(img_array[: , : , 1] , bins=256 , range=(0 , 256))[0]
        hist_b = np.histogram(img_array[: , : , 2] , bins=256 , range=(0 , 256))[0]

        # Calculate the mean intensity for each channel
        mean_r = np.sum(hist_r * np.arange(256)) / np.sum(hist_r)
        mean_g = np.sum(hist_g * np.arange(256)) / np.sum(hist_g)
        mean_b = np.sum(hist_b * np.arange(256)) / np.sum(hist_b)

        # Crop images often have a significant green presence (leaves)
        # Relaxed green dominance: green should be slightly higher than red and blue
        is_green_dominant = mean_g > 50 and mean_g > mean_r + 10 and mean_g > mean_b + 10
        # Allow limited earthy tones for pests, but only if green is still present
        is_earthy = mean_r > 40 and mean_g > 50 and mean_r > mean_b
        is_crop_like = is_green_dominant or is_earthy
        print(
            f"Color Histogram Check - Mean R: {mean_r:.2f}, Mean G: {mean_g:.2f}, Mean B: {mean_b:.2f}, Is Crop-Like: {is_crop_like}")
        return is_crop_like
    except Exception as e:
        print(f"Error in is_crop_like_image: {str(e)}")
        return False


# Calculate edge density for texture analysis
def calculate_edge_density(img):
    try:
        # Convert PIL Image to numpy array and then to grayscale for edge detection
        img_array = np.array(img)
        if len(img_array.shape) != 3:  # Ensure it's an RGB image
            return 0.0
        gray = cv2.cvtColor(img_array , cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray , 100 , 200)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        print(f"Edge Density: {edge_density:.4f}")
        return edge_density
    except Exception as e:
        print(f"Error in calculate_edge_density: {str(e)}")
        return 0.0


# Check if the image is out-of-distribution based on crop model, histogram, and edge density
def is_out_of_distribution(probabilities , img , predicted_class , confidence_threshold=0.1 , entropy_threshold=0.8):
    try:
        probs = probabilities.cpu().numpy().flatten()
        entropy = calculate_entropy(probs)
        max_confidence = np.max(probs)

        # Primary check: Confidence and entropy (stricter thresholds)
        confidence_ood = max_confidence < confidence_threshold
        entropy_ood = entropy > entropy_threshold

        # Secondary check: Color histogram heuristic
        histogram_ood = not is_crop_like_image(img)

        # Tertiary check: Edge density (texture complexity, very permissive threshold)
        edge_density = calculate_edge_density(img)
        edge_ood = edge_density < 1.0  # Extremely permissive threshold

        # Combine checks: Flag as OOD only if confidence/entropy fail OR both histogram and edge density fail
        is_ood = (confidence_ood or entropy_ood) or (histogram_ood and edge_ood)

        # Fallback: If the top predicted class is a pest, override the OOD flag
        is_pest = any(predicted_class in pests[crop] for crop in crops)
        if is_ood and is_pest:
            print(f"Overriding OOD flag: Top predicted class '{predicted_class}' is a pest.")
            is_ood = False

        # Log top-5 predictions for debugging
        top5_indices = np.argsort(probs)[-5:][::-1]
        top5_classes = [(class_names[idx] , probs[idx]) for idx in top5_indices]
        print(f"Top-5 Predictions: {top5_classes}")
        print(
            f"OOD Check Details - Max Confidence: {max_confidence:.4f}, Entropy: {entropy:.4f}, Histogram OOD: {histogram_ood}, Edge OOD: {edge_ood}, Is Pest: {is_pest}, Final Is OOD: {is_ood}")
        return is_ood
    except Exception as e:
        print(f"Error in is_out_of_distribution: {str(e)}")
        return True  # Conservatively flag as OOD if there's an error


# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(filename='app_logs.log' , level=logging.INFO , format='%(asctime)s - %(levelname)s - %(message)s')

# Load the trained crop classification model (ResNet-18)
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    expected_num_classes = len(class_names)  # Should be 27
    model.fc = nn.Sequential(
        nn.Dropout(0.5) ,
        nn.Linear(num_ftrs , 256) ,
        nn.ReLU() ,
        nn.Dropout(0.3) ,
        nn.Linear(256 , expected_num_classes)
    )
    model_path = r"C:\Users\91905\PycharmProjects\AgroSense_AI_v3\app\multi_class_classifier2.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    print(f"Expected number of classes: {expected_num_classes}")
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    print("Crop classification model (ResNet-18) loaded successfully.")
except Exception as e:
    print(f"Error loading crop classification model: {str(e)}")
    raise

# Preload metadata at startup
try:
    preload_metadata()
except Exception as e:
    print(f"Failed to preload metadata: {str(e)}")
    raise

# Load local LLM (distilgpt2) from local directory
try:
    distilgpt2_path = r"C:\Users\91905\PycharmProjects\AgroSense_AI_v3\app\distilgpt2"
    if not os.path.exists(distilgpt2_path):
        raise FileNotFoundError(
            f"distilgpt2 model directory not found at {distilgpt2_path}. Please download the model files.")
    tokenizer = AutoTokenizer.from_pretrained(distilgpt2_path)
    model_llm = AutoModelForCausalLM.from_pretrained(distilgpt2_path)
    model_llm.to(device)
    model_llm.eval()
    print("Local LLM (distilgpt2) loaded successfully from local directory.")
except Exception as e:
    print(f"Error loading local LLM: {str(e)}")
    raise


# Function to apply temperature scaling to logits before softmax
def temperature_scaled_softmax(logits , temperature=2.0):
    scaled_logits = logits / temperature
    return torch.nn.functional.softmax(scaled_logits , dim=1)


# Function to adjust confidence based on damage factor
def adjust_confidence_with_damage(raw_confidence , pred_class):
    # Get the damage factor for the predicted class
    damage_factor = damage_factors.get(pred_class , 0.5)  # Default to 0.5 if not specified

    # Adjust the confidence: scale it by giving more weight to the damage factor
    # New formula: adjusted_confidence = raw_confidence * (0.3 + 0.7 * damage_factor)
    # This gives more influence to high-damage classes
    adjusted_confidence = raw_confidence * (0.3 + 0.7 * damage_factor)

    # Remove the lower cap of 20%, but keep the upper cap at 100%
    adjusted_confidence = min(100.0 , adjusted_confidence)  # Cap at 100%

    print(
        f"Confidence Adjustment - Raw Confidence: {raw_confidence:.2f}%, Damage Factor: {damage_factor}, Adjusted Confidence: {adjusted_confidence:.2f}%")
    return adjusted_confidence


# Function to run model inference with timeout
@timeout(10)  # 10-second timeout
def run_model_inference(model , img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        # Apply temperature scaling to soften probabilities (temperature > 1 makes distribution more uniform)
        probabilities = temperature_scaled_softmax(output , temperature=2.0)
    return output , probabilities


# Refine diagnosis using local LLM with timeout (raw English output)
@timeout(5)  # 5-second timeout
def refine_with_llm(data):
    prompt = (
        f"Refine this crop diagnosis into detailed advice for a farmer, using simple, practical, and actionable language. "
        f"Provide exactly 4 sections: Problem, Why, Fix, and Prevent. "
        f"Each section must have exactly 3 bullet points, and each bullet point must have 2 sentences: the first sentence states the point, and the second sentence explains it in farmer-friendly language. "
        f"Focus on what the farmer can understand and do right now to manage or prevent the issue:\n"
        f"Status: {data['status']}\n"
        f"Message: {data['message']}\n"
        f"Confidence: {data['confidence']}\n"
        f"Severity: {data['severity']}\n"
        f"Details: {data['details']}\n"
        f"Type: {data['type']}\n"
        f"Example format for a section:\n"
        f"Problem:\n"
        f"- Your apple crop has a disease called Apple Scab. This can make your apples look bad and not good for selling.\n"
        f"- It makes black spots on leaves and fruit. This can make your trees weak over time.\n"
        f"- If you don’t act, it can spread to other trees. This will cause a bigger problem for your farm.\n"
        f"Output format:\n"
        f"Problem:\n"
        f"- <Point 1>. <Explanation 1>.\n"
        f"- <Point 2>. <Explanation 2>.\n"
        f"- <Point 3>. <Explanation 3>.\n"
        f"Why:\n"
        f"- <Point 1>. <Explanation 1>.\n"
        f"- <Point 2>. <Explanation 2>.\n"
        f"- <Point 3>. <Explanation 3>.\n"
        f"Fix:\n"
        f"- <Point 1>. <Explanation 1>.\n"
        f"- <Point 2>. <Explanation 2>.\n"
        f"- <Point 3>. <Explanation 3>.\n"
        f"Prevent:\n"
        f"- <Point 1>. <Explanation 1>.\n"
        f"- <Point 2>. <Explanation 2>.\n"
        f"- <Point 3>. <Explanation 3>."
    )
    inputs = tokenizer(prompt , return_tensors="pt" , truncation=True , max_length=500)
    inputs = {k: v.to(device) for k , v in inputs.items()}

    with torch.no_grad():
        outputs = model_llm.generate(
            inputs["input_ids"] ,
            max_new_tokens=300 ,
            min_length=200 ,
            do_sample=True ,
            temperature=0.6 ,
            top_p=0.9 ,
            num_return_sequences=1 ,
            pad_token_id=tokenizer.eos_token_id
        )

    refined_text = tokenizer.decode(outputs[0] , skip_special_tokens=True)
    lines = refined_text.split('\n')
    points = [line.strip() for line in lines if line.strip() and line.strip().startswith('-')]

    # Group points by section
    sections = {"Problem": [] , "Why": [] , "Fix": [] , "Prevent": []}
    current_section = None
    for line in lines:
        line = line.strip()
        if line in ["Problem:" , "Why:" , "Fix:" , "Prevent:"]:
            current_section = line[:-1]  # Remove the colon
        elif line.startswith('-') and current_section:
            sections[current_section].append(line)

    # Validate that we have exactly 3 points per section
    for section , section_points in sections.items():
        if len(section_points) != 3:
            return fallback_refine(data)

    # Flatten the points into a single list for the frontend
    result = []
    for section in ["Problem" , "Why" , "Fix" , "Prevent"]:
        result.append(f"{section}:")
        result.extend(sections[section])

    return result


# Fallback refinement if LLM fails (raw English output)
def fallback_refine(data):
    if data["status"] == "Healthy":
        causes_lines = data["details"]["causes"].split('. ')[:3] if data["details"]["causes"] else [
            "No issues detected." , "Your crop is growing well." , "Keep up the good work."]
        prevention_lines = data["details"]["prevention"].split('. ')[:3] if data["details"]["prevention"] else [
            "Maintain regular watering and fertilization." , "Monitor for early signs of pests or diseases." ,
            "Prune trees to keep them healthy."]
        recommendation_lines = data["details"]["recommendations"].split('. ')[:3] if data["details"][
            "recommendations"] else ["Continue good farming practices." , "Check plants weekly for any changes." ,
                                     "Add compost to soil to keep it rich."]

        problem_points = [
            f"- Your {data['class_name'].split('___')[0].lower()} crop is healthy. This means no diseases or pests are affecting it right now." ,
            f"- We are {data['confidence']} sure of this. You can trust that your crop is doing well." ,
            f"- Your farming practices are working well. Keep doing what you’re doing to maintain this health."
        ]
        why_points = [
            f"- {causes_lines[0]}. Your crop looks strong and free of problems." ,
            f"- {causes_lines[1]}. This shows your care is paying off." ,
            f"- {causes_lines[2]}. Healthy crops mean better yields for you."
        ]
        fix_points = [
            f"- No fixes are needed right now. Your crop is in good shape." ,
            f"- Just keep an eye on your plants. Look for any small changes that might happen." ,
            f"- Stay prepared for weather changes. Rain or heat can sometimes bring new problems."
        ]
        prevent_points = [
            f"- {prevention_lines[0]}. This keeps your plants strong and ready for any challenges." ,
            f"- {prevention_lines[1]}. Catching problems early stops them from getting worse." ,
            f"- {prevention_lines[2]}. Healthy plants grow better and give you more harvest."
        ]
    else:
        class_name = data.get("class_name" , "")
        crop_name = data['class_name'].split('___')[0].lower() if '___' in data['class_name'] else \
            data['class_name'].split('_')[0].lower()
        display_name = class_name.replace(f"{crop_name.capitalize()}___" , "").replace(f"{crop_name.capitalize()}_" ,
                                                                                       "").replace("_" , " ")

        causes_lines = data["details"]["causes"].split('. ')[:3] if data["details"]["causes"] else ["Not available." ,
                                                                                                    "Please check with an expert." ,
                                                                                                    "It may spread in certain weather."]
        prevention_lines = data["details"]["prevention"].split('. ')[:3] if data["details"]["prevention"] else [
            "Keep your farm clean." , "Remove sick plants quickly." , "Water plants carefully."]
        recommendation_lines = data["details"]["recommendations"].split('. ')[:3] if data["details"][
            "recommendations"] else ["Check your plants often." , "Use organic pest control if needed." ,
                                     "Monitor weather conditions."]

        problem_points = [
            f"- Your {crop_name} crop has a {data['type']} called {display_name}. This can harm your plants and reduce your harvest." ,
            f"- It’s {data['severity']} with {data['confidence']} confidence. This means you need to act quickly to save your crop." ,
            f"- This problem can spread to other plants. If you don’t stop it, you might lose more of your crop."
        ]
        why_points = [
            f"- {causes_lines[0]}. This is what’s making your plants sick." ,
            f"- {causes_lines[1]}. Knowing this helps you understand how to stop it." ,
            f"- {causes_lines[2]}. This explains why you’re seeing these signs on your plants."
        ]
        fix_points = [
            f"- {recommendation_lines[0]}. Start doing this today to protect your crop." ,
            f"- {recommendation_lines[1]}. This will help control the problem quickly." ,
            f"- {recommendation_lines[2]}. Keep checking to make sure the problem doesn’t come back."
        ]
        prevent_points = [
            f"- {prevention_lines[0]}. This stops the problem from happening again." ,
            f"- {prevention_lines[1]}. Doing this keeps your plants safe in the future." ,
            f"- {prevention_lines[2]}. This helps your crop stay healthy for the next season."
        ]

    # Combine points into the required format
    result = []
    for section , points in [("Problem" , problem_points) , ("Why" , why_points) , ("Fix" , fix_points) ,
                             ("Prevent" , prevent_points)]:
        result.append(f"{section}:")
        result.extend(points)

    return result


# Define transformation for input image
val_test_transform = transforms.Compose([
    transforms.Resize((224 , 224)) ,
    transforms.ToTensor() ,
    transforms.Normalize(mean=[0.485 , 0.456 , 0.406] , std=[0.229 , 0.224 , 0.225])
])


# Prediction endpoint (returns raw English data)
@app.route('/api/predict' , methods=['POST'])
def predict():
    start_time = time.time()
    success = False
    prediction = None
    confidence = None
    severity = None
    details = None
    refined_details = None
    message = None
    latency = None

    try:
        # Step 0: Log the start of the request
        print("Received prediction request.")

        # Get language from request, default to English
        lang = request.form.get('language' , 'en').lower()
        if lang not in SUPPORTED_LANGUAGES:
            lang = 'en'  # Fallback to English if unsupported language
        print(f"Language selected: {lang}")

        # Step 1: Process the input image
        print("Processing input image...")
        if request.content_type.startswith('multipart/form-data'):  # Handle file upload
            if 'file' not in request.files:
                message = "No file provided in the request."
                raise ValueError(message)
            file = request.files['file']
            if file.filename == '':
                message = "No selected file."
                raise ValueError(message)
            img = process_uploaded_file(file)
        elif request.content_type == 'application/json':  # Handle webcam input
            if 'image' not in request.json:
                message = "No image data provided in JSON."
                raise ValueError(message)
            base64_string = request.json['image']
            img = process_base64_image(base64_string)
            if img is None:
                message = "Failed to process webcam image."
                raise ValueError(message)
        else:
            message = f"Unsupported Content-Type: {request.content_type}"
            raise ValueError(message)

        if img:
            print(
                f"Processing {'webcam' if request.content_type == 'application/json' else 'file'}: {file.filename if 'file' in locals() else 'webcam frame'} with size {img.size}")
            img_tensor = val_test_transform(img).unsqueeze(0).to(device)
            print(f"Input tensor shape: {img_tensor.shape}")

            # Step 2: Run crop classification model with timeout
            print("Running crop classification model...")
            try:
                crop_output , crop_probabilities = run_model_inference(model , img_tensor)
                print(f"Crop model output shape: {crop_output.shape}")
                print(f"Crop probabilities shape: {crop_probabilities.shape}")
            except TimeoutError:
                message = "Model inference timed out. Please try again with a smaller image."
                raise TimeoutError(message)

            # Validate output shape
            if crop_probabilities.shape[1] != len(class_names):
                message = f"Crop model output shape mismatch: expected {len(class_names)} classes, got {crop_probabilities.shape[1]}"
                raise ValueError(message)

            # Log raw probabilities for debugging
            probs = crop_probabilities.cpu().numpy().flatten()
            top5_indices = np.argsort(probs)[-5:][::-1]
            top5_classes = [(class_names[idx] , probs[idx]) for idx in top5_indices]
            print(f"Raw Probabilities (Top-5): {top5_classes}")

            # Step 3: Check if the image is OOD
            print("Checking if image is out-of-distribution...")
            crop_confidence , crop_predicted = torch.max(crop_probabilities , dim=1)
            crop_confidence_value = crop_confidence.item()
            pred_class = class_names[crop_predicted.item()]
            is_ood = is_out_of_distribution(crop_probabilities , img , pred_class)
            print(f"OOD check (crop model): is_ood={is_ood}, crop_confidence={crop_confidence_value:.2f}")

            # Step 4: Handle OOD case
            if is_ood:
                message = "This image does not appear to be a crop-related image."
                latency = (time.time() - start_time) * 1000
                latency_str = f"Latency: {latency:.2f}ms"
                logging.info(
                    f"OOD Detection - Source: {'webcam' if request.content_type == 'application/json' else 'file'}, File: {file.filename if 'file' in locals() else 'N/A'}, Confidence: {crop_confidence_value:.2f}")
                return jsonify({
                    "message": message ,
                    "latency": latency_str ,
                    "debug_info": f"Predicted Class: {pred_class}, Confidence: {crop_confidence_value:.2f}"
                })

            # Step 5: Proceed with crop prediction if not OOD
            print("Proceeding with crop prediction...")
            raw_confidence = crop_confidence_value * 100
            # Adjust confidence based on damage factor
            confidence = adjust_confidence_with_damage(raw_confidence , pred_class)
            severity = determine_severity(confidence)
            pred_class = class_names[crop_predicted.item()]
            crop = determine_crop_from_class(pred_class)
            if crop == "unknown":
                message = f"Could not determine crop for predicted class: {pred_class}"
                raise ValueError(message)

            # Log whether the prediction is a pest or disease
            issue_type = "pest" if pred_class in pests[crop] else "disease" if pred_class in diseases[
                crop] else "healthy"
            print(
                f"Prediction output: {pred_class}, type: {issue_type}, confidence: {confidence:.2f}%, severity: {severity}")

            metadata = metadata_dict.get(normalize_class_name(pred_class) ,
                                         {"causes": "" , "prevention": "" , "recommendations": ""})
            farm_output = format_farm_output(pred_class , crop , metadata , f"{confidence:.2f}%" , severity)
            prediction = pred_class
            details = farm_output

            # Step 6: Refine with LLM (returns raw English data)
            print("Refining prediction with LLM...")
            try:
                refined_details = refine_with_llm(farm_output)
            except TimeoutError:
                print("LLM refinement timed out.")
                refined_details = fallback_refine(farm_output)
            success = True

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Prediction error: {str(e)}\nStack trace:\n{error_trace}")
        message = str(e) if str(
            e) else "An unexpected error occurred while processing the image. Please ensure the image is valid and try again."
        logging.error(f"Prediction error: {str(e)}\nStack trace:\n{error_trace}")

    latency = (time.time() - start_time) * 1000
    latency_str = f"Latency: {latency:.2f}ms"
    logging.info(
        f"Prediction - Success: {success}, Latency: {latency:.2f}ms, Source: {'webcam' if request.content_type == 'application/json' else 'file'}, File: {file.filename if 'file' in locals() else 'N/A'}, Prediction: {prediction if prediction else 'N/A'}, Message: {message if message else 'None'}")

    if success:
        return jsonify({
            "prediction": prediction ,
            "confidence": f"{confidence:.2f}%" ,
            "severity": severity ,
            "details": refined_details ,
            "latency": latency_str
        })
    else:
        return jsonify({"message": message , "latency": latency_str})


# Translation endpoint
@app.route('/api/translate' , methods=['POST'])
def translate():
    try:
        data = request.get_json()
        if not data or 'texts' not in data or 'language' not in data:
            return jsonify({"error": "Invalid request. Must provide 'texts' and 'language'."}) , 400

        texts = data['texts']
        lang = data['language'].lower()
        if lang not in SUPPORTED_LANGUAGES:
            lang = 'en'  # Fallback to English
        translated_texts = batch_translate_texts(texts , lang)
        return jsonify({"translated_texts": translated_texts})
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return jsonify({"error": str(e)}) , 500


@app.route('/' , methods=['GET' , 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                return jsonify({"status": "uploaded" , "filename": file.filename})
        return jsonify({"status": "error" , "message": "No file uploaded"})
    return render_template('index.html')


if __name__ == '__main__':
    host = '127.0.0.1'
    port = 5000
    print(f"Starting Flask app on http://127.0.0.1:{port}/")
    app.run(host=host , port=port , debug=True , use_reloader=False)