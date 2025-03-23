import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import time
from scipy import stats

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPUs Available:")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        details = tf.config.experimental.get_device_details(device)
        device_name = details.get("device_name", "")
        print(f"Device: {device}, Details: {details}")
        if "4070" in device_name:
            print("Primary GPU is RTX4070")
else:
    print("No GPU available. Using CPU.")

IMG_SIZE = (224, 224)
BATCH_SIZE = 128
EPOCHS = 20
original_dataset_dir = 'data/xray_new'
classes = ['Abnormal', 'Normal']

all_data, all_labels = [], []
for class_name in classes:
    class_dir = os.path.join(original_dataset_dir, class_name)
    all_data.extend([os.path.join(class_dir, fname) for fname in os.listdir(class_dir)])
    all_labels.extend([class_name] * len(os.listdir(class_dir)))

train_data, temp_data, train_labels, temp_labels = train_test_split(
    all_data, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
val_data, test_data, val_labels, test_labels = train_test_split(
    temp_data, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

train_df = pd.DataFrame({'filename': train_data, 'class': train_labels})
val_df = pd.DataFrame({'filename': val_data, 'class': val_labels})
test_df = pd.DataFrame({'filename': test_data, 'class': test_labels})

train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_dataframe(train_df, x_col='filename', y_col='class',
                                                    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
validation_generator = val_test_datagen.flow_from_dataframe(val_df, x_col='filename', y_col='class',
                                                            target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
test_generator = val_test_datagen.flow_from_dataframe(test_df, x_col='filename', y_col='class',
                                                      target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

model_list = [
    ("MobileNetV2", tf.keras.applications.MobileNetV2),
    ("VGG16", tf.keras.applications.VGG16),
    ("VGG19", tf.keras.applications.VGG19),
    ("ResNet50", tf.keras.applications.ResNet50),
    ("InceptionV3", tf.keras.applications.InceptionV3),
    ("Xception", tf.keras.applications.Xception),
    ("DenseNet121", tf.keras.applications.DenseNet121),
    ("NASNetMobile", tf.keras.applications.NASNetMobile),
    ("NASNetLarge", tf.keras.applications.NASNetLarge)
]

results = []

for model_name, model_fn in model_list:
    print(f"Training {model_name}...")
    base_model = model_fn(include_top=False, input_shape=(*IMG_SIZE, 3), weights='imagenet')
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(classes), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(train_generator, validation_data=validation_generator, epochs=EPOCHS, verbose=1)
    training_time = time.time() - start_time
    print(f"Training time for {model_name}: {training_time:.2f} seconds")

    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes

    n = len(y_true)
    standard_error = np.sqrt((test_accuracy * (1 - test_accuracy)) / n)
    confidence_interval = stats.norm.interval(0.95, loc=test_accuracy, scale=standard_error)
    print(f"{model_name} Test Accuracy: {test_accuracy:.4f}")
    print(f"{model_name} 95% Confidence Interval: {confidence_interval}")

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    def plot_training_history(history, model_name):
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Training and Validation Accuracy - {model_name}')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss - {model_name}')
        plt.legend()
        plt.show()

    plot_training_history(history, model_name)
    report = classification_report(y_true, y_pred_classes, target_names=classes, output_dict=True)
    accuracy = report['accuracy']
    results.append((model_name, test_accuracy, accuracy, confidence_interval))

results_df = pd.DataFrame(results, columns=['Model', 'Test Accuracy', 'Classification Report Accuracy', 'Confidence Interval'])
print(results_df)

plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Test Accuracy', data=results_df)
plt.title('Model Test Accuracy Comparison')
plt.xticks(rotation=45)
plt.show()
