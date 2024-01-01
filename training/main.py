import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121, VGG19, ResNet101V2, InceptionResNetV2, MobileNet, InceptionV3
from tensorflow.keras.layers import *
from tensorflow.keras import layers, models, Model
from tensorflow.keras.callbacks import EarlyStopping,CSVLogger, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.backend import *
import seaborn as sns
from datetime import datetime


tf.keras.backend.clear_session()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Load CSV file with image names and labels
df = pd.read_csv('./dataset/lable.csv')


batch_size=8
# Define the data generator for preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values
    validation_split=0.2 , # Split dataset into training and validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )
size=256
# Create training and validation generators
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='./dataset/',
    x_col='IMG',
    y_col=['GLM','MNG','NTM','PIT'],
    subset='training',
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    class_mode='raw',  # Use 'raw' for 1-hot encoding
    target_size=(size, size),
    color_mode='grayscale'  
)

valid_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='./dataset/',
     x_col='IMG',
    y_col=['GLM','MNG','NTM','PIT'],
    subset='validation',
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    class_mode='raw',
    target_size=(size, size),
    color_mode='grayscale'
)

for data_batch, labels_batch in train_generator:
    print('Data batch shape:', data_batch.shape)
    print('Labels batch shape:', labels_batch.shape)
    break

name_model='Test_MobileNet'
# from models import InceptionNet
model=MobileNet(weights=None,input_shape=(size,size,1),classes=4)
for layers in model:
    layers.trainable=False
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model_checkpoint = ModelCheckpoint(f"./trained models/train_{name_model}.h5", save_best_only=True)

early_stopping=tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=12,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=True,
)
history = model.fit(train_generator,epochs=50,validation_data=valid_generator, batch_size=batch_size, shuffle=True,callbacks=[model_checkpoint, early_stopping])
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title(f"{name_model} Results", fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.show()

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f"{name_model}_results")
plt.legend(loc='lower right')
plt.show()
hist_df = pd.DataFrame(history.history) 

hist_csv_file = f'./training_results/{name_model}_history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
print('created csv file')
print(f'cell executeted on :{datetime.now()}')