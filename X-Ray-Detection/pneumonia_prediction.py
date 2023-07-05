# for os operations
import os

from dotenv import load_dotenv

# for matrix operations
import numpy as np

# for dataframes
import pandas as pd

# for random number generation
import secrets

# for machine learning utilities
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score

# for deep learning stuffs
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50

print(tf.__version__)

load_dotenv()

model_weight = os.environ.get("model")

BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
IM_SIZE_W = int(os.environ.get('IM_SIZE_w'))
IM_SIZE_H = int(os.environ.get('IM_SIZE_H'))

AUTOTUNE = tf.data.experimental.AUTOTUNE

tf.random.set_seed(10)

# Define CNN model
def create_model():
    with tf.device('/gpu:0'):        
    
        # Model input
        input_layer = layers.Input(shape=(IM_SIZE_W, IM_SIZE_H, 1), name='input')  
        
        # First block
        x = layers.Conv2D(filters=128, kernel_size=3, 
                          activation='relu', padding='same', 
                          name='conv2d_1')(input_layer)
        x = layers.MaxPool2D(pool_size=2, name='maxpool2d_1')(x)
        x = layers.Dropout(0.1, name='dropout_1')(x)

        # Second block
        x = layers.Conv2D(filters=128, kernel_size=3, 
                          activation='relu', padding='same', 
                          name='conv2d_2')(x)
        x = layers.MaxPool2D(pool_size=2, name='maxpool2d_2')(x)
        x = layers.Dropout(0.1, name='dropout_2')(x)

        # Third block
        x = layers.Conv2D(filters=128, kernel_size=3, 
                          activation='relu', padding='same', 
                          name='conv2d_3')(x)
        x = layers.MaxPool2D(pool_size=2, name='maxpool2d_3')(x)
        x = layers.Dropout(0.1, name='dropout_3')(x)

        # Fourth block
        x = layers.Conv2D(filters=256, kernel_size=3, 
                          activation='relu', padding='same', 
                          name='conv2d_4')(x)
        x = layers.MaxPool2D(pool_size=2, name='maxpool2d_4')(x)
        x = layers.Dropout(0.1, name='dropout_4')(x)

        # Fifth block
        x = layers.Conv2D(filters=256, kernel_size=3, 
                          activation='relu', padding='same', 
                          name='conv2d_5')(x)
        x = layers.MaxPool2D(pool_size=2, name='maxpool2d_5')(x)
        x = layers.Dropout(0.1, name='dropout_5')(x)

        # Sixth block
        x = layers.Conv2D(filters=512, kernel_size=3, 
                          activation='relu', padding='same', 
                          name='conv2d_6')(x)
        x = layers.MaxPool2D(pool_size=2, name='maxpool2d_6')(x)
        x = layers.Dropout(0.1, name='dropout_6')(x)

        # Seventh block
        x = layers.Conv2D(filters=512, kernel_size=3, 
                          activation='relu', padding='same', 
                          name='conv2d_7')(x)
        x = layers.MaxPool2D(pool_size=2, name='maxpool2d_7')(x)
        x = layers.Dropout(0.1, name='dropout_7')(x)
        
        # GlobalAveragePooling
        x = layers.GlobalAveragePooling2D(name='global_average_pooling2d')(x)   
        x = layers.Flatten()(x)
        
        # Head
        x = layers.Dense(1024,activation='relu')(x)
        x = layers.Dropout(0.1, name='dropout_head_2')(x)
        x = layers.Dense(128,activation='relu')(x)
        
        # Output
        output = layers.Dense(units=4, 
                              activation='softmax', 
                              name='output')(x)


        model = Model(input_layer, output)
    
        
        

        F_1_macro = tfa.metrics.f_scores.F1Score(num_classes=4, average="macro", name='f1_macro') 
        
        model.compile(optimizer='adam', 
                      loss='categorical_crossentropy', 
                      metrics=[F_1_macro, 'accuracy'])

    return model



def pre_process_image(img):
    classes = ['Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral', 'COVID-19']
    df = pd.DataFrame()
    df['filename'] = [img]
    df['class'] = [secrets.choice(classes)]
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_gen = test_datagen.flow_from_dataframe(df,
                                        x_col="filename",
                                        y_col="class",
                                        target_size=(IM_SIZE_W, IM_SIZE_H),
                                        color_mode='grayscale',
                                        batch_size=BATCH_SIZE,
                                        class_mode='categorical',
                                        shuffle=False,
                                        num_parallel_calls=AUTOTUNE)
    return test_gen
    

def predict_probability(img):

    test_gen = pre_process_image(img)
    model = create_model()
    model.load_weights(model_weight)
    predict = model.predict(test_gen)

    prd = np.argmax(predict)
    lab = np.max(predict)
    class_dict = {
        0 : 'Normal',
        1 : 'Pneumonia-Bacterial',
        2 : 'Pneumonia-Viral',
        3 : 'COVID-19',
    }
    res = f'Predicted with {lab:.3f} probability that the X-Ray is {class_dict[prd]}'
    return res  

