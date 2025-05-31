# 👋 Hi, I'm Hira Batool Zaidi  
Data Scientist and AI Engineer passionate about building impactful AI solutions in Computer Vision, NLP, and predictive modeling.


---

## 🧠 About Me  
- 💡 I specialize in solving real-world problems using AI & ML  
- 💄 Built a smart lipstick color detector using computer vision  
- 🧠 Focused on Deep Learning, Computer Vision, and NLP  
- 📊 Strong in data analysis, model training, and deployment  
- 🛠️ Always exploring new ways to blend AI with creativity  

---

## 💼 Projects


### 💄 Lipstick Shade Detector & Recommender  
Detects lipstick color from lips in real-time using face landmarks and suggests matching branded shades (with name, number, price).  
Supports celebrity image input and provides real-brand lipstick matches.  
**Key Features:**  
- Facial landmark detection using `MediaPipe`  
- Lip region extraction  
- Lip color detection using HSV & CNN  
- Lipstick brand matcher using scraped catalog  
- ResNet-based model for shade classification  
- Deployed using `Gradio` interface for real-time demo  

**Tech Stack:** `Python`, `MediaPipe`, `OpenCV`, `NumPy`, `HSV`, `CNN`, `ResNet`, `Gradio`
Lipstick Detector Demo : ![Black Color Result](https://github.com/user-attachments/assets/18afd0ba-04d2-40b9-8f70-0d9d39568154)
                         ![red color result](https://github.com/user-attachments/assets/e3a18818-482e-4e8d-bea5-9f93cf6608ed)


### 🧵 Fabric Color Variation Classifier  
Automated color classification and quality control of fabric images using custom JSON labeling and K-Nearest Neighbors (KNN) classification.  
- Created JSON files for dataset annotation manually  
- Automated fabric color labeling based on HSV thresholds  
- Classified images as Pass/Fail based on similarity thresholds using KNN  
- Handled large datasets with extensive automation scripts for color extraction and labeling
- 

**Tech Stack:** `Python`, `OpenCV`, `NumPy`, `Pandas`, `KNN`, `HSV`, `JSON`
### 🧩 Fabric Color Detection Flowchart

![Fabric Color Flowchart](![image](https://github.com/user-attachments/assets/6a16f025-6037-415f-ba58-1b9585fc5b09)
)


---

### 📊 Stock Price Prediction  
Forecasts short-term and long-term stock trends using time-series models and deep learning.  
**Tech Stack:** `Pandas`, `LSTM`, `ARIMA`, `Prophet`, `TensorFlow`,sklearn, yfinance
![Stock Prediction Result](![image](https://github.com/user-attachments/assets/f557d866-756f-4125-a0a0-c4756cd9690e)
)


---
***Sentiment Analysis Project***
Project Goal
The goal of this project is to develop a machine learning model capable of classifying the sentiment of text data (such as reviews or social media posts) into categories like positive, negative, or neutral. This helps in understanding public opinion, customer feedback, and improving decision-making processes.

**Overview***
Data Loading: Loaded a dataset containing text samples and their corresponding sentiment labels.
Text Preprocessing: Cleaned the text by converting to lowercase, removing punctuation, and eliminating stopwords using the NLTK library.
Feature Extraction: Used TF-IDF Vectorizer to transform text data into numerical feature vectors capturing the importance of words.
Model Training: Trained a Logistic Regression model on the vectorized features to classify sentiments.
Evaluation: Tested the model on unseen data and evaluated performance using classification metrics such as precision, recall, and F1-score.
Model Persistence: Saved the trained model and vectorizer using joblib for later use.
***Usage**
test_text = "I really love this product!"
print("Sentiment:", predict_sentiment(test_text))  # Expected output: Positive
test_text = "This is the worst service ever."
print("Sentiment:", predict_sentiment(test_text))  # Expected output: Negative


Prediction Function: Created a function that takes raw text input, cleans and vectorizes it, and predicts the sentiment label.
---
### 📄 Word to PDF Converter
A Python-based web app that converts Microsoft Word documents (.docx) into PDF files.
Built using Gradio for the interface and LibreOffice for document conversion in headless mode.
Supports easy upload and download via browser.

Tech Stack: Python, Gradio, LibreOffice
For best performance, it is recommended to upload files up to 50 MB in size.
Below is a screenshot of the Word to PDF Converter interface.
Upload a .docx file and download the converted PDF easily.
![image](https://github.com/user-attachments/assets/55fe0c5c-6e8a-4afe-9767-5fbe5e3be819)
![image](https://github.com/user-attachments/assets/04480531-3d51-4916-beed-d402af637b28)


🍼***Water Bottle Detection Model***
Developed an object detection model to identify water bottles and similar objects in images and videos with high accuracy and efficiency.
The model is built using deep learning techniques and can be used for real-time detection in automation processes for quality control and retail environments.
Includes a training script (train.py) to train or fine-tune the model on custom datasets.
Pre-trained model weights are provided (model.pt) for immediate use in inference tasks without needing retraining.
Easily extensible for detecting other similar objects by updating the training dataset.

**Tech Stack***: Python, PyTorch (or TensorFlow if applicable), OpenCV, Deep Learning, CNN, Object Detection


## 🛠️ Skills  
`Python` `Pandas` `NumPy` `Matplotlib` `Seaborn`  
`OpenCV` `MediaPipe` `Gradio` `scikit-learn` `TensorFlow` `Keras`  
`CNN` `ResNet` `FNN` `KNN` `YOLO` `HSV` `JSON` `Transformers`  
`LSTM` `ARIMA` `Prophet` `NLP` `EDA`  
`Flask` `Streamlit` `Git`  
 

---

## 📫 Let's Connect  
- 🔗 [LinkedIn](https://www.linkedin.com/in/syeda-hira-batool-zaidi-9824a922a/)  
- 📧 Email: syedhirabatool512@gmail.com
