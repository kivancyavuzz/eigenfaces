# PCA-Based Face Recognition (Eigenfaces)

This project implements a complete face recognition pipeline using **Principal Component Analysis (PCA)** and the **Eigenface** approach. It explores data preprocessing, dimensionality reduction, face reconstruction, and classification robustness under noisy conditions.

---

## 🚀 Key Features

### 1. Data Preprocessing & Normalization
* Loads and flattens face images into vectors.
* Computes the "Mean Face" of the dataset and performs zero-mean normalization to highlight unique facial features.

### 2. PCA & Eigenface Computation
* Efficiently computes eigenvalues and eigenvectors using the covariance matrix.
* Generates **Eigenfaces** (the principal components of the face space) and visualizes the variance captured by each component.

### 3. Face Reconstruction
* Projects original face images into the lower-dimensional eigenface space.
* Reconstructs faces using $M$ principal components and calculates **Mean Squared Error (MSE)** to analyze reconstruction quality.

### 4. Recognition & Classification
* Implements a **Nearest Neighbor (NN)** classifier using **Leave-One-Out** cross-validation.
* Analyzes recognition accuracy relative to the number of eigenfaces used ($M$).

### 5. Noise Robustness Analysis
* Tests the classifier's performance against **Gaussian** and **Salt-and-Pepper** noise.
* Visualizes the drop in accuracy as noise levels increase, demonstrating the model's limits.

---

## 🛠️ Technical Stack
* **Language:** Python
* **Libraries:** `NumPy`, `Matplotlib`, `Pillow (PIL)`, `Scikit-learn`, `Seaborn`.

## 📈 Usage
The script uses command-line arguments for flexible data handling:

```bash
python eigenfaces.py --data_path ./path_to_data --output_path ./results
