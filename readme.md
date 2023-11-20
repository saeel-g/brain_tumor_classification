
# NeuroScan: Classifying Brain Tumor MRI Images
In the realm of medical imaging, NeuroScan is a deep learning model designed to classify Brain Tumor MRI images. Trained on various CNN architectures, including DenseNet121, InceptionV3, InceptionResNetV2, and MobileNet, NeuroScan achieved the highest accuracy with InceptionResNetV2 (96.65%). Consequently, InceptionResNetV2 was selected for implementation in the application, showcasing its effectiveness in accurate tumor classification.

## Features
* **InceptionResNetV2 Architecture:** NeuroScan utilizes the InceptionResNetV2 neural network architecture, combining inception modules and residual connections for intricate feature extraction and information flow.

* **Four-Class Classification:** The model classifies Brain Tumor MRI images into four distinct classes, facilitating targeted medical interventions.

* **Enhanced Accuracy and Sensitivity:** NeuroScan achieves high accuracy in tumor classification with sensitivity to subtle abnormalities, ensuring adaptability to varying image characteristics.

## How it Works:
* **Image Preprocessing:**
MRI images undergo preprocessing, including resizing, normalization, and grayscale conversion.

* **Inference and Prediction:**
NeuroScan uses learned features to predict the presence and type of brain tumor, aiding medical professionals in decision-making.

## Getting Started

To run this application locally, follow these steps:

1. Clone the GitHub repository:
```bash
git clone https://github.com/saeel-g/ChatPDF.git
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. create a ".env" file in the same project folder and inside it write:
```bash
OPENAI_API_KEY= Insert your key here
```
4. Insert following command in terminal
```bash
streamlit run main.py
```
5. Open your web browser and navigate to given local host url to use the app.

# Usage
1. Upload a PDF file using the "Upload PDF" button.
2. Ask questions in the chat interface.
3. The chatbot will analyze the PDF content and provide answers.

# Support
If you encounter any issues or have questions, please create a GitHub issue

# Contact
For any inquiries, you can contact us at saeelg30@gmail.com.