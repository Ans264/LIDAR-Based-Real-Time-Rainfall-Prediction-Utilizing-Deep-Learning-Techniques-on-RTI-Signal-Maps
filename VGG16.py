import os
import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
from keras.applications import EfficientNetB0
from keras.layers import GlobalAveragePooling2D
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input


# Directories containing signal maps
rain_dir = "C:/Users/ANSUMAN/Downloads/NARL-Fig/Rainfall"
likely_rain_dir = "C:/Users/ANSUMAN/Downloads/NARL-Fig/Likely"
no_rain_dir = "C:/Users/ANSUMAN/Downloads/NARL-Fig/No_Rainfall"

# Function to load and preprocess images
def load_and_preprocess_images(directory, crop_size=(464, 319)):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Crop image to the desired size
            height, width = img.shape
            start_row = (height - crop_size[1]) // 2
            start_col = (width - crop_size[0]) // 2
            cropped_img = img[start_row:start_row + crop_size[1], start_col:start_col + crop_size[0]]
            # Normalize the image
            scaler = MinMaxScaler()
            cropped_img = scaler.fit_transform(cropped_img)
            # Apply Gaussian filter for noise reduction
            cropped_img = cv2.GaussianBlur(cropped_img, (5, 5), 0)
            images.append(cropped_img)
    return np.array(images)

# Load and preprocess images
rain_images = load_and_preprocess_images(rain_dir)
likely_rain_images = load_and_preprocess_images(likely_rain_dir)
no_rain_images = load_and_preprocess_images(no_rain_dir)

# Display some standardized images
def display_images(images, title):
    plt.figure(figsize=(10, 5))
    for i in range(min(5, len(images))):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()

display_images(rain_images, "Rain")
display_images(likely_rain_images, "Likely to Rain")
display_images(no_rain_images, "No Rain")

# Data augmentation: rotation and flipping
def augment_images(images):
    augmented_images = []
    for img in images:
        # Rotate images
        rows, cols = img.shape
        for angle in [90, 180, 270]:
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            rotated_img = cv2.warpAffine(img, M, (cols, rows))
            augmented_images.append(rotated_img)
        # Flip images
        flipped_img_h = cv2.flip(img, 1)  # Horizontal flip
        flipped_img_v = cv2.flip(img, 0)  # Vertical flip
        augmented_images.append(flipped_img_h)
        augmented_images.append(flipped_img_v)
    return np.array(augmented_images)

# Augment images
rain_images = augment_images(rain_images)
likely_rain_images = augment_images(likely_rain_images)
no_rain_images = augment_images(no_rain_images)

# Combine and label data
X = np.concatenate((rain_images, likely_rain_images, no_rain_images), axis=0)
y = np.array([0] * len(rain_images) + [1] * len(likely_rain_images) + [2] * len(no_rain_images))

# Convert labels to categorical (one-hot encoding)
y = to_categorical(y, num_classes=3)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the data for CNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

# Define CNN model for spatial feature extraction
#cnn_model = Sequential([
    #Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    #MaxPooling2D(pool_size=(2, 2)),
    #Conv2D(64, (3, 3), activation='relu'),
    #MaxPooling2D(pool_size=(2, 2)),
    #Conv2D(128, (3, 3), activation='relu'),
    #MaxPooling2D(pool_size=(2, 2)),
    #Flatten(),  # Ensures a fixed number of features
    #Dense(256, activation='relu'),
    #Dense(3, activation='softmax')  # Three output classes
#])

# Compile and train the CNN model
#cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#history = cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.25)

# Evaluate the model
#y_pred = np.argmax(cnn_model.predict(X_test), axis=1)
#y_true = np.argmax(y_test, axis=1)
#print(classification_report(y_true, y_pred, target_names=['Rain', 'Likely to Rain', 'No Rain']))



# Load VGG16 with ImageNet weights, exclude top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(X_train.shape[1], X_train.shape[2], 3))

# Convert grayscale to 3-channel before feeding to VGG16
X_train_vgg = np.repeat(X_train, 3, axis=-1)
X_test_vgg = np.repeat(X_test, 3, axis=-1)

# Freeze the base VGG16 layers
for layer in base_model.layers:
    layer.trainable = False

# Add a new classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)
vgg_model = Model(inputs=base_model.input, outputs=predictions)

# Compile and train
vgg_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = vgg_model.fit(X_train_vgg, y_train, epochs=10, batch_size=32, validation_split=0.25)

# Evaluate
y_pred = np.argmax(vgg_model.predict(X_test_vgg), axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred, target_names=['Rain', 'Likely to Rain', 'No Rain']))







# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.show()

# Display some predictions
def display_predictions(images, true_labels, predicted_labels, class_names):
    plt.figure(figsize=(15, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"True: {class_names[true_labels[i]]}\nPred: {class_names[predicted_labels[i]]}")
        plt.axis('off')
    plt.show()

# Get a subset of test images for displaying predictions
test_images = X_test_vgg[:10].reshape(-1, X_test_vgg.shape[1], X_test_vgg.shape[2])
display_predictions(test_images, y_true[:10], y_pred[:10], ['Rain', 'Likely to Rain', 'No Rain'])

# Display incorrect predictions
def display_incorrect_predictions(images, true_labels, predicted_labels, class_names):
    incorrect_indices = np.where(true_labels != predicted_labels)[0]
    if len(incorrect_indices) == 0:
        print("No incorrect predictions to display.")
        return

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(incorrect_indices[:10]):  # Display up to 10 incorrect predictions
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[idx], cmap='gray')
        plt.title(f"True: {class_names[true_labels[idx]]}\nPred: {class_names[predicted_labels[idx]]}")
        plt.axis('off')
    plt.show()

# Display incorrect predictions
display_incorrect_predictions(X_test_vgg.reshape(-1, X_test_vgg.shape[1], X_test_vgg.shape[2]), y_true, y_pred, ['Rain', 'Likely to Rain', 'No Rain'])

# Histograms of predicted probabilities
def plot_predicted_probabilities(X_test_vgg, y_test, vgg_model):
    y_pred_prob = vgg_model.predict(X_test_vgg)
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.hist(y_pred_prob[:, i], bins=20, color='blue', alpha=0.7)
        plt.title(f'Predicted Probabilities for Class {i}')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
    plt.show()

plot_predicted_probabilities(X_test_vgg, y_test, vgg_model)

# Confusion matrix heatmap
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Rain', 'Likely to Rain', 'No Rain'], yticklabels=['Rain', 'Likely to Rain', 'No Rain'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# Save the trained model
#cnn_model.save('rainfall_prediction_model.h5')