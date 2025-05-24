AgroSense AI
AgroSense AI is an intelligent agricultural tool that leverages deep learning for crop health classification and NLP-based support using DistilGPT2. It diagnoses the health of three crops—apples, potatoes, and tomatoes—by identifying healthy crops, diseases, and pests.
📂 Project Structure

app/ - Backend logic and API endpoints for the web application
code/ - Scripts for data preprocessing, model training, and inference
templates/ - HTML templates for the frontend interface
translations/ - Language files for multilingual support
.gitignore - Excludes large files (e.g., datasets, models) from GitHub
requirements.txt - Lists project dependencies

❗ Excluded from GitHub
Due to size constraints, the following are not included in the repository:

Model files (.onnx, .pth)
Full datasets
Logs and virtual environments

📊 Dataset
The custom dataset is curated from three Kaggle sources, focusing on apples, potatoes, and tomatoes:

New Plant Diseases Dataset - Healthy and diseased crop images - link: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
PlantVillage Dataset - Healthy and diseased crop images - link: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
IP02 Dataset - Pest images - link: https://www.kaggle.com/datasets/rtlmhjbn/ip02-dataset

Dataset Structure

Crops: Apples, Potatoes, Tomatoes
Splits: Train, Valid, Test
Train/Valid: Contain crops and pests folders
crops/healthy/: Images of healthy crops
crops/diseased/: Images of diseases (e.g., Apple_Scab, Potato_Late_Blight)
pests/: Images of pests (e.g., Aphids, Fruit_Fly) common across crops


Test: Contains images without subfolders for healthy/diseased/pests


Pests: Four common pests (e.g., Aphids, Fruit_Fly) across all crops

🚀 Features

Crop Health Classification: Uses a ResNet18 model to classify 27 conditions (healthy, diseases, pests) with high accuracy.
Farmer-Friendly Outputs: Provides actionable insights (causes, prevention, recommendations) in multiple languages.
Multilingual Support: Frontend interface supports language switching via i18next and translated metadata.

🛠️ Setup

Clone the repository: git clone <repo-url>
Install dependencies: pip install -r requirements.txt
Download datasets from the Kaggle links above and update base_path in the scripts.
Run the training script: python code/train.py
Launch the web app: python app/main.py

📈 Usage

Access the web interface via the provided URL.
Upload a crop image (PNG/JPG/JPEG) of apples, potatoes, or tomatoes.
View the diagnosis (healthy, disease, or pest) and metadata (causes, prevention, recommendations).
Switch languages using the dropdown menu for localized outputs.

📜 License
This project is licensed under the MIT License.
📧 Contact
For issues or inquiries, open a GitHub issue or contact the maintainers.
