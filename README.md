# 🥔 Potato Leaf Disease Classification using CNN

This project is based on **Lab Sheet 05** from the CCS4310 – Deep Learning course. The aim is to build a Convolutional Neural Network (CNN) model that classifies potato leaf images into three categories:

- **Early Blight**
- **Late Blight**
- **Healthy**

---

## 📁 Dataset

Download the dataset from the [Google Drive link](https://drive.google.com/drive/folders/1wCBW5OiokzhPXgtyXrC3A8CeqYJw6zR?usp=sharing).

Dataset is organized into directories by class label.

---

## 🎯 Objectives

- Perform dataset exploration and visualization.
- Preprocess the images using `ImageDataGenerator`.
- Build and train a CNN model using Keras.
- Evaluate the model and improve its accuracy.
- Save and load the model for future inference.
- Test the model with new real images.

---

## 🧪 Steps Implemented

### ✅ Problem 1: Dataset Exploration
- Loaded dataset using `ImageDataGenerator`.
- Printed number of images per category.
- Visualized sample images using `matplotlib`.

### ✅ Problem 2: Image Preprocessing
- Rescaled and resized images (128x128).
- Applied augmentation: rotation, zoom, horizontal flip.
- Split data into training and validation sets.

### ✅ Problem 3: Model Building
- Built a CNN model with:
  - 2+ Conv2D layers
  - MaxPooling
  - Dropout layer
  - Flatten and Dense layers
- Used ReLU and Softmax activation.

### ✅ Problem 4: Model Compilation and Training
- Compiled with `categorical_crossentropy` loss and `adam` optimizer.
- Trained model with 15+ epochs.
- Plotted training vs validation accuracy/loss.

### ✅ Problem 5: Model Evaluation
- Evaluated accuracy on test set.
- Plotted confusion matrix.
- Displayed classification report using `sklearn`.

### ✅ Problem 6: Save and Load Model
- Saved model using `model.save()`.
- Reloaded model and used `predict()` on a single image.

### ✅ Problem 7: Real Image Testing
- Preprocessed 3 external images manually.
- Predicted their classes with the trained model.

### ⚙️ Problem 8: Accuracy Improvement
- Added more Conv2D + MaxPooling layers.
- Increased training epochs.
- Applied BatchNormalization.
- Tuned learning rate and batch size.

---

## 📊 Results

- Final accuracy: **XX.XX%** (update with actual)
- Model performs well on unseen data and real images.

---

## 🛠️ Tools & Libraries

- Python 3.x
- TensorFlow / Keras
- NumPy, Matplotlib
- scikit-learn
- Google Colab (recommended)

---

## 📸 Sample Output

Include example accuracy/loss plots and a confusion matrix here as images (if uploading to repo).

---

## 📝 References

- Lab Material and Dataset Provided
- [YouTube Playlist - CNN Tutorial](https://www.youtube.com/watch?v=dGtDTjYs3xc&list=PLeo1K3hjS3ut49PskOfLnE6WUoOp_2lsD)

---

## 👤 Author

- **Name**: Vayoni Gamage  
- **Index Number**: [Your SLTC Index Here]  
- **Course**: CCS4310 – Deep Learning

---

## 📌 How to Run

```bash
# 1. Clone this repo
git clone https://github.com/YourUsername/Potato-Leaf-CNN.git
cd Potato-Leaf-CNN

# 2. (Optional) Create virtual environment and activate
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the notebook or script
