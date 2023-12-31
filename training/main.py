import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet201, VGG19, ResNet101V2, InceptionResNetV2, MobileNet, InceptionV3
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
df = pd.read_csv('../dataset/lable.csv')


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
    directory='../dataset/',
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
    directory='../dataset/',
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

name_model='DenseNet201'
# from models import InceptionNet
base_model=DenseNet201(weights='imagenet', include_top=False, input_shape=(size,size,1))
for layer in base_model.layers:
    layer.trainable=False

x=GlobalAveragePooling2D()(base_model.output)
x=Dense(1024, activation='relu')(x)
x=BatchNormalization()(x)
x=Dropout(0.1)(x)
x=Dense(512, activation='relu')(x)
x=BatchNormalization()(x)
# x=Dense(256, activation='relu')(x)
output=Dense(4, activation='softmax')(x)

model=Model(inputs=base_model.input, outputs=output)
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model_checkpoint = ModelCheckpoint(f"../trained models/train_{name_model}.h5", save_best_only=True, verbose=1)


history = model.fit(train_generator,epochs=50,validation_data=valid_generator, batch_size=batch_size, shuffle=True,callbacks=[model_checkpoint])


hist_df = pd.DataFrame(history.history) 

hist_csv_file = f'./training_results/{name_model}_history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
print('created csv file')
print(f'cell executeted on :{datetime.now()}')