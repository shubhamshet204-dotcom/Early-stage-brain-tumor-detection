# Advanced Early-Stage Brain Tumor Detection & Classification

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Supported-red?logo=keras)](https://keras.io)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org)

An end-to-end deep learning project utilizing Convolutional Neural Networks (CNN) to detect and classify brain tumors from Magnetic Resonance Imaging (MRI) scans. The model is built using TensorFlow and Keras to classify MRI scans into four distinct categories with high accuracy.

---

## 📂 Project Directory Structure

Here is the structure of the project repository. Note that the downloaded dataset folder and compiled model weights are ignored from git to keep the repository lightweight.

```text
EarlyStage_brain_tumor_detection/
│
├── .gitignore                       # Rules to prevent committing datasets, models, and cache
├── README.md                        # Documentation of the project
├── brain_tumor_clg_project (2).ipynb# Core Jupyter Notebook containing the training pipeline
│
└── brain-tumor-classification-mri/  # Ignored dataset folder (automatically downloaded via notebook)
    ├── Training/                    # 2,870 MRI images used for model training and validation
    │   ├── glioma_tumor/            # Glioma MRI scans
    │   ├── meningioma_tumor/        # Meningioma MRI scans
    │   ├── no_tumor/                # Healthy brain MRI scans
    │   └── pituitary_tumor/         # Pituitary MRI scans
    │
    └── Testing/                     # 394 MRI images used for model testing
        ├── glioma_tumor/
        ├── meningioma_tumor/
        ├── no_tumor/
        └── pituitary_tumor/
```

---

## ⚡ Key Project Features

* **Multi-Class Classification:** Classifies MRI scans into 4 target categories (3 tumor types and 1 healthy type), avoiding simple binary classification limitations.
* **Optimized Image Ingestion:** Utilizes `ImageDataGenerator` to load, rescale, and resize images on-the-fly, managing memory footprint efficiently.
* **High-Resolution Preprocessing:** Images are processed at `150x150` resolution (upgraded from `64x64`) to capture subtle structural tumor abnormalities.
* **Overfitting Mitigation:** Incorporates `Dropout` regularization and validation monitoring to ensure generalization.
* **Strict Evaluation Protocol:** Set `shuffle=False` on the test generator to guarantee that predicted classes map correctly to the true classes in the confusion matrix and classification report.

---

## 🛠️ Step-by-Step Pipeline Walkthrough

### Step 1: Data Acquisition & Preprocessing
The dataset is downloaded directly from Kaggle and preprocessed. Pixel values are rescaled from `[0, 255]` to `[0, 1]` using `ImageDataGenerator`:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Rescale pixel values
datagen = ImageDataGenerator(rescale=1./255)

# Load training dataset
train_generator = datagen.flow_from_directory(
    '/content/brain-tumor-classification-mri/Training',
    target_size=(150, 150),
    batch_size=32,
    classes=['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor'],
    class_mode='categorical',
    shuffle=True
)
```

---

### Step 2: Model Design (Custom CNN)
The custom CNN architecture extracts spatial features from MRI scans using convolutional layer blocks before classifying them via dense layers:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    # Convolution Block 1
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),

    # Convolution Block 2
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    # Convolution Block 3
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    # Fully Connected Blocks
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # 50% neurons deactivated to prevent overfitting
    Dense(4, activation='softmax') # 4 units and softmax output for multiclass probabilities
])
```

---

### Step 3: Compilation & Training
The model is compiled with the **Adam optimizer** and **Categorical Crossentropy** loss function, which tracks training metrics over epochs:

```python
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10
)
```

---

### Step 4: Metric Evaluation
Evaluation is executed on the test dataset. **Important:** Setting `shuffle=False` aligns the predictions index-by-index with true test classes:

```python
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Ensure shuffle=False in test_generator before evaluation
y_true = test_generator.classes
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)

# Print metrics
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=class_labels))
```

---

### Step 5: Real-World Single Image Inference
You can upload an individual MRI scan and get a prediction with confidence levels:

```python
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

class_labels = ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']

# Load and preprocess sample image
img = image.load_img('/content/no tumor.jpg', target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Classify
predictions = model.predict(img_array)[0]
predicted_idx = np.argmax(predictions)
predicted_class = class_labels[predicted_idx]
confidence = predictions[predicted_idx] * 100

if predicted_class == 'no_tumor':
    print(f"❌ Result: No Tumor Detected ({confidence:.2f}% confidence)")
else:
    print(f"✅ Result: Tumor Detected ({predicted_class}) ({confidence:.2f}% confidence)")
```

---

## ⚙️ Environment Setup & Run Guide

### Running in Google Colab (Recommended)
1. Upload `brain_tumor_clg_project (2).ipynb` to [Google Colab](https://colab.research.google.com/).
2. Enable GPU acceleration: **Runtime** > **Change runtime type** > Select **T4 GPU**.
3. Create a Kaggle API token to authenticate the dataset download:
   * Go to Kaggle Settings > click **Create New Token** to download `kaggle.json`.
   * Open the JSON file and copy your `username` and `key` into the prompt when downloading the dataset.
4. Run all cells sequentially.

### Running Locally (Jupyter Lab / Desktop)
1. Clone the project locally:
   ```bash
   git clone https://github.com/shubhamshet204-dotcom/Early-stage-brain-tumor-detection.git
   cd Early-stage-brain-tumor-detection
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow opencv-python matplotlib scikit-learn opendatasets numpy
   ```
3. Open the Jupyter interface and run the notebook:
   ```bash
   jupyter notebook
   ```

---

## 🛡️ License
This project is open-source under the MIT License.
