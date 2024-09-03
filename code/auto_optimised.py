# %%
# %pip install keras-tuner
# %pip install tensorflow-addons  # For extra metrics

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3, MobileNetV3Small
from tensorflow.keras import layers, models
from keras_tuner import HyperModel
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa  # Additional metrics
import os
import numpy as np

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Define the hypermodel
class WasteClassificationHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build(self, hp):
        # Choose the model
        model_type = hp.Choice('model_type', ['InceptionV3', 'MobileNetV3', 'DenseNet201'])
        if model_type == 'InceptionV3':
            base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif model_type == 'DenseNet201':
            base_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False, input_shape=self.input_shape)
        else:
            base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        # Freeze the base model
        base_model.trainable = False
        
        # Define the model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'),
            layers.Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)),
            layers.Dense(self.num_classes, activation='softmax', dtype='float32')  # Final layer with float32 precision
        ])
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
            loss='categorical_crossentropy',
            metrics=['accuracy', tfa.metrics.F1Score(num_classes=self.num_classes, average='macro'), 'Precision', 'Recall']
        )
        
        return model


# %%
# Define the data generator with augmentation and train-validation-test split
def create_data_generators(hp, dataset_directory):
    augmentation_technique = hp.Choice('augmentation_technique', ['none', 'flip', 'rotation', 'zoom'])
    
    if augmentation_technique == 'flip':
        datagen = ImageDataGenerator(horizontal_flip=True)
    elif augmentation_technique == 'rotation':
        datagen = ImageDataGenerator(rotation_range=20)
    elif augmentation_technique == 'zoom':
        datagen = ImageDataGenerator(zoom_range=0.2)
    else:
        datagen = ImageDataGenerator()

    # Split the data into train, validation, and test sets
    train_generator = datagen.flow_from_directory(
        os.path.join(dataset_directory, 'train'),
        target_size=(224, 224),
        batch_size=hp.Int('batch_size', min_value=16, max_value=64, step=16),
        class_mode='categorical'
    )

    validation_generator = datagen.flow_from_directory(
        os.path.join(dataset_directory, 'validation'),
        target_size=(224, 224),
        batch_size=hp.Int('batch_size', min_value=16, max_value=64, step=16),
        class_mode='categorical'
    )

    test_generator = datagen.flow_from_directory(
        os.path.join(dataset_directory, 'test'),
        target_size=(224, 224),
        batch_size=hp.Int('batch_size', min_value=16, max_value=64, step=16),
        class_mode='categorical'
    )

    return train_generator, validation_generator, test_generator


# %%
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import multi_gpu_model

# Redirect console output to a file
log_file = 'console_output_small.log'
sys.stdout = open(log_file, 'w')

# Define a custom callback to capture accuracy at each epoch
class AccuracyLogger(Callback):
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file
        # Open the log file in write mode
        with open(self.log_file, 'w') as f:
            f.write('Epoch,Accuracy,Validation Accuracy\n')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        accuracy = logs.get('accuracy')
        val_accuracy = logs.get('val_accuracy')

        # Write the epoch, accuracy, and validation accuracy to the log file
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch+1},{accuracy},{val_accuracy}\n")

# Define the dataset directory and number of classes
dataset_directory = '/app/data/Datasets/Trashnet-resized'  # Update this path as necessary
num_classes = 6  # Set the number of classes

# Input shape
input_shape = (224, 224, 3)

# log file for accuracy
log_file = 'accuracy_log1.csv'

# Create the hypermodel
hypermodel = WasteClassificationHyperModel(input_shape, num_classes)

# Define the tuner
tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='tuning_results',
    project_name='waste_classification'
)

# Define the search space and perform tuning
train_generator, validation_generator, test_generator = create_data_generators(tuner.oracle.hyperparameters, dataset_directory)
tuner.search(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[AccuracyLogger(log_file), EarlyStopping(patience=2)]
)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Print the summary of the best model
best_model.summary()

# Evaluate the model on the test set
test_loss, test_accuracy, test_f1, test_precision, test_recall = best_model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test F1 Score: {test_f1}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")

# Close the log file to ensure all output is written
sys.stdout.close()

