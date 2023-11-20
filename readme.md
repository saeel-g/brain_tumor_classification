
# NeuroScan: Classifying Brain Tumor MRI Images
In the realm of medical imaging, NeuroScan is a deep learning model designed to classify Brain Tumor MRI images. Trained on various CNN architectures, including DenseNet121, InceptionV3, InceptionResNetV2, and MobileNet, NeuroScan achieved the highest accuracy with InceptionResNetV2 (96.65%). Consequently, InceptionResNetV2 was selected for implementation in the application, showcasing its effectiveness in accurate tumor classification.

## Features
* **InceptionResNetV2 Architecture:** NeuroScan utilizes the InceptionResNetV2 neural network architecture, combining inception modules and residual connections for intricate feature extraction and information flow.

* **Four-Class Classification:** The model classifies Brain Tumor MRI images into four distinct classes, facilitating targeted medical interventions.

* **Enhanced Accuracy and Sensitivity:** NeuroScan achieves high accuracy in tumor classification with sensitivity to subtle abnormalities, ensuring adaptability to varying image characteristics.

## How it Works:
### **Training**
* **Image Preprocessing & Augmentation:**
MRI images undergo preprocessing, including resizing, rescaling along with data augmentation using the ImageDataGenerator of Tensorflow
* **Data Generators:**
Using the ImageDataGenerator, training and validation data generators (train_generator and valid_generator) are created. These generators are used to flow images and labels from the DataFrame directory, applying data augmentation and normalization.

* **Model Architecture:** 
The code defines the models using TensorFlow's Keras API. The model is compiled with the Adam optimizer and categorical crossentropy loss. Checkpoints are saved during training to monitor progress.

* **Training the Model:**
The model is trained using the fit function. Training occurs for 50 epochs with early stopping based on validation accuracy. The training history is stored in a CSV file, and plots of accuracy and loss are displayed using Matplotlib.

* **Results Visualization:**
The code generates and displays two plots:
The first plot shows training and validation accuracy over epochs.
The second plot shows training loss and validation accuracy over epochs.
CSV File Creation:
The training history (accuracy and loss over epochs) is stored in a CSV file for further analysis.

## Deployment
The Brain Tumor MRI Images Classifier is deployed using the Streamlit library, offering an interactive and user-friendly interface for image classification.

Access the Deployed Application
The deployed application is accessible through the following link:

[Brain Tumor MRI Images Classifier]()
### How to Use
* **Upload Image:** Use the file uploader to select a Brain Tumor MRI image in JPG, PNG, or JPEG format.

* **Prediction:** Once the image is uploaded, the application provides a real-time prediction, indicating the predicted class of the brain tumor.

* **Interpret Results:** The predicted class is displayed, allowing users to interpret the model's classification. 

## Local Setup Guide
To run this application locally, follow these steps:

1. Clone the GitHub repository:
```bash
git clone https://github.com/saeel-g/brain_tumor_classification
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Insert following command in terminal
```bash
streamlit run main.py
```
4. Open your web browser and navigate to given local host url to use the app.

# Support
If you encounter any issues or have questions, please create a GitHub issue

# Contact
For any inquiries, you can contact saeelg30@gmail.com.