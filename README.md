# Traffic-Signs-Recognition-Project
Traffic Sign Recognition Project- Self Driving Car

A deep learning-based project for recognizing German traffic signs using a Convolutional Neural Network (CNN) inspired by LeNet architecture. This project demonstrates data preprocessing, augmentation, model training, and evaluation on the German Traffic Sign Recognition Benchmark (GTSRB) dataset.


Overview

This project implements a traffic sign classifier using TensorFlow. Key features include:

Image preprocessing (grayscale conversion, normalization, histogram equalization).
Data augmentation (rotation, translation, shear, zoom) to handle class imbalance.
LeNet-based CNN model with dropout for regularization.
Training on augmented data with validation and testing splits.
Visualization of dataset distribution, augmented samples, and training progress.

The model achieves high accuracy on the test set, making it suitable for real-world applications like autonomous driving assistance systems.

Dataset

The project uses the German Traffic Sign Dataset from Kaggle, containing ~39,000 training images across 43 classes. The dataset is stored in german-traffic-sign-dataset.zip (handled via Git LFS for large files).

Training set: 39,209 images (augmented to ~382,789).
Validation set: 4,410 images.
Test set: 12,630 images.

Data is loaded from pickle files (train.p, valid.p, test.p) and preprocessed for model input.
Installation

Clone the repository:textgit clone https://github.com/shakir2018/Traffic-Signs-Recognition-Project.git
cd Traffic-Signs-Recognition-Project
Install Git LFS (for large dataset files):textapt-get install git-lfs
git lfs install
git lfs pull

Install dependencies (Python 3.6+ recommended; tested on Google Colab with TensorFlow 1.x):textpip install -r requirements.txt(Note: If not present, create requirements.txt with: tensorflow==1.15.0, numpy, matplotlib, opencv-python, scikit-learn, etc.)
For Kaggle dataset download (optional, if not using the zipped file):
Place your kaggle.json in the root.
Run: kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

Usage

Run the Notebook:

Open traffic_Signs_Recognition_Project.ipynb in Jupyter or Google Colab.
Execute cells sequentially to load data, preprocess, train the model, and evaluate.

Interactive Demo:

Use the Binder badge above to run the notebook in-browser (no setup needed).
Or view statically via nbviewer.

Training:text# In the notebook, set EPOCHS, BATCH_SIZE, dropout, and run the training loop.Model checkpoints are saved in the models/ folder.
Prediction:
Use the trained model to predict on new images via the predict_tf1 function.


Model Architecture

The model is a modified LeNet CNN:

Input: 32x32x1 grayscale images.
Layers:
Conv2D (5x5, 6 filters) → MaxPool → Conv2D (5x5, 16 filters) → MaxPool.
Flatten → Dense (120) → Dense (84) → Dense (43, softmax).

Optimizer: Adam.
Loss: Categorical Cross-Entropy.
Dropout: 0.3 during training.
Batch Size: 128.
Epochs: 150.

Training and Evaluation

Data Augmentation: 10 augmented versions per original image (rotation, shift, shear, zoom).
Evaluation: Accuracy on training (~92.7%), validation (~92.5%), and test sets.
Logging: Training progress saved in training.log.

Example output:
textEPOCH 150 - 1969 sec ...
Training accuracy = 0.924 Validation accuracy = 0.925
Visualizations (in Plots/):

Class distribution.
Random samples.
Augmented images.
Transformation examples.

Results

Final Test Accuracy: ~92% (varies with runs; see notebook for details).
Confusion matrix and top errors available in the notebook.
Model generalizes well due to augmentation, handling real-world variations like lighting and angles.


Contributing

Contributions are welcome! Please:

Fork the repo.
Create a feature branch.
Commit changes.
Push and open a Pull Request.

Issues for bugs or enhancements are appreciated.

License

This project is licensed under the MIT License - see the LICENSE file for details.

