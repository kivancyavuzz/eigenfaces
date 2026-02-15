#Kıvanç Yavuz 150230050

# Libraries

# In[ ]:


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import seaborn as sns
# Task 1:
# Data Preprocessing
# In[ ]:
# Load all images and flatten them into vectors
def load_images(data_path, img_size=(112, 92)):
    images = []
    labels = []
    for person in sorted(os.listdir(data_path)):
        person_folder = os.path.join(data_path, person)
        if os.path.isdir(person_folder):
            for img_file in sorted(os.listdir(person_folder)):
                path = os.path.join(person_folder, img_file)
                img = Image.open(path).resize(img_size)
                img_array = np.asarray(img, dtype=np.float32).T.flatten()
                images.append(img_array)
                labels.append(person)
    return np.array(images), labels

# Subtract mean face to normalize
def normalize_faces(images):
    avg_face = np.mean(images, axis=0)
    adjusted_images = images - avg_face
    return adjusted_images, avg_face

def dump_image(img_vector, shape, filename):
    img = img_vector.reshape(shape)
    img = np.transpose(img)
    img = np.clip(img, 0, 255)
    img_scaled = img / 255.0
    plt.imshow(img_scaled, cmap='gray')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

# Run task 1: load data, normalize, save mean faces
def run_task1(data_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    shape = (112, 92)
    images, labels = load_images(data_path, shape)
    X_all, mean_all = normalize_faces(images)
    dump_image(mean_all, shape, os.path.join(output_path, 'mean_face_all.png'))
    first_10 = images[:10]
    _, mean_10 = normalize_faces(first_10)
    dump_image(mean_10, shape, os.path.join(output_path, 'mean_face_subset.png'))
    return X_all, mean_all, labels

# Task 2:
# PCA and Eigenface Computation
# In[ ]:


def pca_analysis(X):
    covariance = np.dot(X, X.T)
    e_vals, e_vecs = np.linalg.eigh(covariance)
    transformed_vecs = np.dot(X.T, e_vecs)
    transformed_vecs /= np.linalg.norm(transformed_vecs, axis=0)
    idx = np.argsort(e_vals)[::-1]
    e_vals = e_vals[idx]
    transformed_vecs = transformed_vecs[:, idx]
    return e_vals, transformed_vecs

# Visualization Functions

# In[ ]:


def show_top_faces(eigenfaces, shape, M, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(M):
        ef = eigenfaces[:, i]
        img = ef.reshape(shape)
        img = np.transpose(img)
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0
        img = np.clip(img, 0, 255)
        img_scaled = img / 255.0
        plt.imshow(img_scaled, cmap='gray')
        plt.axis('off')
        plt.title(f"Eigenface {i+1}")
        plt.savefig(os.path.join(output_dir, f"ef_{i+1}.png"), bbox_inches='tight')
        plt.close()

def visualize_evals(eigvals, output_path):
    plt.plot(eigvals, marker='o')
    plt.title("Eigenvalues (variance)")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid()
    plt.savefig(os.path.join(output_path, "eigenvalues.png"))
    plt.close()

def cumulative_variance_graph(eigvals, output_path):
    cum_var = np.cumsum(eigvals) / np.sum(eigvals)
    plt.plot(cum_var, marker='o')
    plt.title("Cumulative Variance")
    plt.xlabel("Num Eigenfaces")
    plt.ylabel("Cumulative Variance")
    plt.grid()
    plt.savefig(os.path.join(output_path, "cumulative_variance.png"))
    plt.close()


# Main TASK 2 Runner

# In[ ]:


def run_task2(X, shape, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    eigvals, eigvecs = pca_analysis(X)

    show_top_faces(eigvecs, shape, 10, os.path.join(output_path, "eigenfaces"))

    visualize_evals(eigvals, output_path)
    cumulative_variance_graph(eigvals, output_path)

    return eigvals, eigvecs


# Task 3:

# Reconstruction Function

# In[ ]:


def recreate_face(face_vector, avg_face, eigenfaces, M):
    weights = np.dot(eigenfaces[:, :M].T, face_vector - avg_face)
    recon = avg_face + np.dot(eigenfaces[:, :M], weights)
    return recon


# MSE Calculation and Visual Record

# In[ ]:


def run_task3(X, avg_face, eigenfaces, labels, output_path, shape):
    idxs = [i for i in range(len(labels)) if labels[i] in ["s10", "s11"]]

    # Make output folder if it doesn’t exist
    if not os.path.exists(os.path.join(output_path, "reconstructed")):
        os.makedirs(os.path.join(output_path, "reconstructed"))

    mse_log = []

    for i in idxs:
        original = X[i] + avg_face

        for M in [10, 20, 50, 100, 200, 300]:
            recon = recreate_face(original, avg_face, eigenfaces, M)
            mse = mean_squared_error(original, recon)
            mse_log.append(f"{labels[i]}_M{M}: {mse:.2f}")

            # Show original and reconstructed side by side
            fig, axs = plt.subplots(1, 2)
            original_img = original.reshape(shape)
            original_img = np.transpose(original_img)
            original_img = np.clip(original_img, 0, 255) / 255.0
            axs[0].imshow(original_img, cmap='gray')
            axs[0].set_title("Original")
            axs[0].axis('off')

            recon_img = recon.reshape(shape)
            recon_img = np.transpose(recon_img)
            recon_img = np.clip(recon_img, 0, 255) / 255.0
            axs[1].imshow(recon_img, cmap='gray')
            axs[1].set_title(f"Recon M={M}")
            axs[1].axis('off')

            # Save figure
            fname = f"comparison_{labels[i]}_M{M}.png"
            plt.savefig(os.path.join(output_path, "reconstructed", fname))
            plt.close()

    # Save MSE results to file
    with open(os.path.join(output_path, "reconstructed", "mse_reconstruction.txt"), "w") as f:
        for line in mse_log:
            f.write(line + "\n")


# Task 4

# Project Faces into Eigenface Space

# In[ ]:


def embed_into_space(X, avg_face, eigenfaces, M):
    return np.dot(X - avg_face, eigenfaces[:, :M])


# Nearest Neighbor Classification

# In[ ]:


def recognize_faces(X_proj, labels, M_values, eigenfaces, avg_face, output_path):
    unique_labels = sorted(set(labels))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_index[lbl] for lbl in labels])

    os.makedirs(os.path.join(output_path, "recognition"), exist_ok=True)
    accuracies = []

    for M in M_values:
        features = embed_into_space(X_proj + avg_face, avg_face, eigenfaces, M)

        # Leave-One-Out Nearest Neighbor
        predictions = []
        for i in range(len(features)):
            query = features[i]
            train = np.delete(features, i, axis=0)
            train_labels = np.delete(numeric_labels, i)

            dists = np.linalg.norm(train - query, axis=1)
            nearest_idx = np.argmin(dists)
            predictions.append(train_labels[nearest_idx])

        acc = accuracy_score(numeric_labels, predictions)
        accuracies.append(acc)

        # Save Confusion Matrix
        cm = confusion_matrix(numeric_labels, predictions)
        plt.figure(figsize=(8, 8))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix (M={M})")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(os.path.join(output_path, "recognition", f"confusion_matrix_M{M}.png"))
        plt.close()

    return accuracies, M_values


# Plot Accuracy vs. Number of Eigenfaces

# In[ ]:


def accuracy_vs_M_graph(accuracies, M_values, output_path):
    plt.plot(M_values, accuracies, marker='o')
    plt.title("Recognition Accuracy vs. Number of Eigenfaces")
    plt.xlabel("Number of Eigenfaces (M)")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(os.path.join(output_path, "recognition", "accuracy_vs_eigenfaces.png"))
    plt.close()


# Run TASK 4: Classification Pipeline

# In[ ]:


def run_task4(X, avg_face, eigenfaces, labels, output_path):
    M_values = [10, 20, 50, 100, 200, 300]
    accuracies, M_vals = recognize_faces(X, labels, M_values, eigenfaces, avg_face, output_path)
    accuracy_vs_M_graph(accuracies, M_vals, output_path)


# Task 5:

# Add Gaussian or Salt-and-Pepper Noise

# In[ ]:


def add_gaussian_noise(image, std_dev):
    noise = np.random.normal(0, std_dev, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255)

def add_salt_and_pepper_noise(image, amount=0.05):
    noisy = image.copy()
    num_salt = int(amount * image.size * 0.5)
    num_pepper = int(amount * image.size * 0.5)

    salt_coords = np.random.randint(0, image.size, num_salt)
    pepper_coords = np.random.randint(0, image.size, num_pepper)

    flat = noisy.flatten()
    flat[salt_coords] = 255
    flat[pepper_coords] = 0
    return flat.reshape(image.shape)


# Evaluate Classification Accuracy on Noisy Images

# In[ ]:


def test_with_noise(X, avg_face, eigenfaces, labels, output_path, noise_type="gaussian", M=50):
    os.makedirs(os.path.join(output_path, "noise"), exist_ok=True)
    selected_indices = np.random.choice(len(X), 10, replace=False)
    accuracies = []
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    label_to_index = {label: i for i, label in enumerate(sorted(set(labels)))}
    numeric_labels = np.array([label_to_index[lbl] for lbl in labels])

    features_clean = embed_into_space(X + avg_face, avg_face, eigenfaces, M)

    for level in noise_levels:
        X_noisy = X.copy()

        for i in selected_indices:
            original = X[i] + avg_face
            if noise_type == "gaussian":
                noisy = add_gaussian_noise(original, std_dev=level * 50)
            elif noise_type == "salt_pepper":
                noisy = add_salt_and_pepper_noise(original, amount=level)
            else:
                raise ValueError("Unsupported noise type.")

            X_noisy[i] = noisy - avg_face

            # Save noisy image
            dump_image(noisy, (112, 92), os.path.join(output_path, "noise", f"noisy_{noise_type}_{level}_{i}.png"))

        features_noisy = embed_into_space(X_noisy + avg_face, avg_face, eigenfaces, M)

        predictions = []
        for i in range(len(features_noisy)):
            query = features_noisy[i]
            train = np.delete(features_clean, i, axis=0)
            train_labels = np.delete(numeric_labels, i)

            dists = np.linalg.norm(train - query, axis=1)
            nearest_idx = np.argmin(dists)
            predictions.append(train_labels[nearest_idx])

        acc = accuracy_score(numeric_labels, predictions)
        accuracies.append(acc)

    return noise_levels, accuracies


# Plot Accuracy vs. Noise Level

# In[ ]:


def noise_accuracy_graph(noise_levels, accuracies, output_path, noise_type="gaussian"):
    plt.plot(noise_levels, accuracies, marker='o')
    plt.title(f"Recognition Accuracy vs. {noise_type.capitalize()} Noise")
    plt.xlabel("Noise Level")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(os.path.join(output_path, "noise", f"accuracy_vs_noise_{noise_type}.png"))
    plt.close()


# Run TASK 5

# In[ ]:


def run_task5(X, avg_face, eigenfaces, labels, output_path):
    for noise_type in ["gaussian", "salt_pepper"]:
        levels, accs = test_with_noise(X, avg_face, eigenfaces, labels, output_path, noise_type=noise_type)
        noise_accuracy_graph(levels, accs, output_path, noise_type=noise_type)


# In[ ]:


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    X, avg_face, labels = run_task1(args.data_path, args.output_path)
    eigvals, eigvecs = run_task2(X, (112, 92), args.output_path)
    run_task3(X, avg_face, eigvecs, labels, args.output_path, (112, 92))
    run_task4(X, avg_face, eigvecs, labels, args.output_path)
    run_task5(X, avg_face, eigvecs, labels, args.output_path)