import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import time
from scipy import stats

# GPU Configuration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("No GPU available. Using CPU.")

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-4
DROPOUT_RATES = [0.5, 0.3]

# SwiGLU Activation Function
def swiGLU(x):
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
    return tf.keras.activations.swish(x1) * x2

# Attention Mechanism
def channel_attention(input_feature, ratio=8):
    channels = input_feature.shape[-1]
    shared_layer_one = tf.keras.layers.Dense(channels // ratio, activation='relu')
    shared_layer_two = tf.keras.layers.Dense(channels)
    avg_pool = shared_layer_two(shared_layer_one(tf.keras.layers.GlobalAveragePooling2D()(input_feature)))
    max_pool = shared_layer_two(shared_layer_one(tf.keras.layers.GlobalMaxPooling2D()(input_feature)))
    attention = tf.keras.layers.Activation('sigmoid')(tf.keras.layers.Add()([avg_pool, max_pool]))
    return tf.keras.layers.multiply([input_feature, attention])

# Data Preparation
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

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, zoom_range=0.1, horizontal_flip=True)
val_test_datagen = ImageDataGenerator(rescale=1./255)

desired_class_order = ['Normal', 'Abnormal']
train_generator = train_datagen.flow_from_dataframe(train_df, x_col='filename', y_col='class', target_size=IMG_SIZE,
                                                    batch_size=BATCH_SIZE, class_mode='binary', classes=desired_class_order)
validation_generator = val_test_datagen.flow_from_dataframe(val_df, x_col='filename', y_col='class', target_size=IMG_SIZE,
                                                            batch_size=BATCH_SIZE, class_mode='binary', classes=desired_class_order)
test_generator = val_test_datagen.flow_from_dataframe(test_df, x_col='filename', y_col='class', target_size=IMG_SIZE,
                                                      batch_size=BATCH_SIZE, class_mode='binary', classes=desired_class_order, shuffle=False)

def create_attention_mlp_model(base_model_fn):
    base_model = base_model_fn(include_top=False, input_shape=(*IMG_SIZE, 3), weights='imagenet')
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs)
    x = channel_attention(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128 * 2)(x)
    x = tf.keras.layers.Lambda(swiGLU)(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATES[1])(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

model = create_attention_mlp_model(tf.keras.applications.MobileNetV2)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_path = 'weights/best_model.h5'
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)

start_time = time.time()
history = model.fit(train_generator, validation_data=validation_generator, epochs=EPOCHS, verbose=1,
                    callbacks=[early_stopping, model_checkpoint])
training_time = time.time() - start_time

model.load_weights(checkpoint_path)
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
y_pred = model.predict(test_generator).ravel()
y_pred_classes = (y_pred >= 0.5).astype(int)
y_true = test_generator.classes

n = len(y_true)
std_err = np.sqrt((test_accuracy * (1 - test_accuracy)) / n)
ci = stats.norm.interval(0.95, loc=test_accuracy, scale=std_err)

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_true, y_pred_classes), annot=True, fmt='d', cmap='viridis',
            xticklabels=desired_class_order, yticklabels=desired_class_order)
plt.title("Confusion Matrix")
plt.show()

fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

results_csv = pd.DataFrame({'filename': test_generator.filenames, 'true_class': y_true,
                            'predicted_class': y_pred_classes, 'abnormal_prob': y_pred})
results_csv.to_csv('outputs/attention_mlp_predictions.csv', index=False)

print(f"Best Model Test Accuracy: {test_accuracy:.4f}")
print(f"95% Confidence Interval: ({ci[0]:.4f}, {ci[1]:.4f})")
