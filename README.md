# Early-Stage Brain Tumor Detection & Classification

This repository contains a deep learning project utilizing Convolutional Neural Networks (CNNs) to detect and classify brain tumors from Magnetic Resonance Imaging (MRI) scans. The model is built using TensorFlow and Keras.

---

## 📌 Project Overview

Brain tumors are one of the most critical healthcare concerns, and early detection plays a pivotal role in successful treatment planning. This project classifies brain MRI scans into four distinct categories:
1. **Glioma Tumor**
2. **Meningioma Tumor**
3. **Pituitary Tumor**
4. **No Tumor** (Healthy Brain)

The pipeline features a multi-class CNN classification architecture, optimized image generators, performance evaluation metrics (confusion matrices, classification reports), and a ready-to-use prediction interface for single MRI scans.

---

## 📊 Dataset Details

The model is trained on the **Brain Tumor Classification (MRI)** dataset sourced from Kaggle:
* **Dataset Link:** [Kaggle - Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
* **Training Images:** 2,870 images across 4 classes
* **Testing Images:** 394 images across 4 classes

---

## 🛠️ Model Architecture

The custom CNN model consists of the following sequential layers:
* **Conv2D Layer (32 filters, 3x3 kernel, ReLU activation)** + **MaxPooling2D (2x2)**
* **Conv2D Layer (64 filters, 3x3 kernel, ReLU activation)** + **MaxPooling2D (2x2)**
* **Conv2D Layer (128 filters, 3x3 kernel, ReLU activation)** + **MaxPooling2D (2x2)**
* **Flatten Layer** (Converts 2D feature maps to a 1D vector)
* **Dense Layer (128 units, ReLU activation)**
* **Dropout Layer (50% rate)** (Prevents overfitting)
* **Dense Output Layer (4 units, Softmax activation)** (Produces probabilities for each class)

---

## 🚀 How to Run the Project

### Option A: Open in Google Colab (Recommended)
1. Upload `brain_tumor_clg_project (2).ipynb` to your [Google Colab](https://colab.research.google.com/).
2. Enable GPU acceleration: Go to **Runtime** > **Change runtime type** > Select **T4 GPU**.
3. Create a Kaggle API token to download the dataset:
   * Go to your Kaggle Account Settings and click **Create New Token** to download `kaggle.json`.
   * Open the file and copy your `username` and `key` into the notebook's prompt when downloading the dataset.
4. Run all cells sequentially.

### Option B: Local Execution
1. Clone this repository:
   ```bash
   git clone https://github.com/shubhamshet204-dotcom/Early-stage-brain-tumor-detection.git
   cd Early-stage-brain-tumor-detection
   ```
2. Install the required Python packages:
   ```bash
   pip install tensorflow opencv-python matplotlib scikit-learn opendatasets numpy
   ```
3. Update the Colab-specific paths (like `/content/`) in the notebook to point to your local directories.
4. Run the notebook using Jupyter:
   ```bash
   jupyter notebook
   ```

---

## 📈 Evaluation & Results

The training pipeline compiles the model with the **Adam optimizer** and **Categorical Crossentropy loss**, generating the following outcomes:
* **Training and Validation Accuracy Curves** plotted automatically to visualize model learning.
* **Confusion Matrix** to inspect classification details per class.
* **Classification Report** yielding metrics such as Precision, Recall, and F1-score for Glioma, Meningioma, Pituitary, and Healthy MRI scans.

---

## 🔬 Single Image Inference Example

The project includes an inference helper function to predict individual MRI images:
```python
# Predict an uploaded MRI image
predictions = model.predict(preprocessed_image)[0]
predicted_idx = np.argmax(predictions)
predicted_class = class_labels[predicted_idx]
confidence = predictions[predicted_idx] * 100

print(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)")
```
* Returns `❌ Result: No Tumor Detected` if healthy.
* Returns `✅ Result: Tumor Detected (<Tumor Type>)` if a tumor is present, along with the prediction confidence.

---

## 📝 License
This project is open-source and available under the MIT License.
