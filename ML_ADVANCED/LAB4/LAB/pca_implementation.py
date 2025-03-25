import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt('./ML_ADVANCED/LAB4/LAB/data/data_pca.txt')
print(f"Data shape: {data.shape}")

# PCA Implementation From Scratch
class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        
    def fit(self, X):
        # 1. Compute the mean of the dataset
        self.mean = np.mean(X, axis=0)
        
        # 2. Center the data
        X_centered = X - self.mean
        
        # 3. Compute the covariance matrix
        # Since the data points are rows, we need to adjust the formula slightly
        n_samples = X.shape[0]
        cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
        
        # 4. Compute eigenvectors and eigenvalues of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvectors by decreasing eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 5. Select the top P principal components
        if self.n_components is None:
            self.n_components = X.shape[1]
        
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        
        return self
    
    def transform(self, X):
        # 6. Project the data onto principal components
        X_centered = X - self.mean
        X_transformed = X_centered @ self.components
        return X_transformed
    
    def inverse_transform(self, X_transformed):
        # 7. Reconstruct the data in the original space
        X_reconstructed = X_transformed @ self.components.T + self.mean
        return X_reconstructed

# Create and fit PCA with full components
pca_full = PCA(n_components=2)
pca_full.fit(data)

# Get the transformed data
data_transformed_full = pca_full.transform(data)
print(f"Transformed data shape: {data_transformed_full.shape}")

# Calculate explained variance ratio
explained_var_ratio = pca_full.explained_variance / np.sum(pca_full.explained_variance)
print("Explained variance ratio by component:")
for i, ratio in enumerate(explained_var_ratio):
    print(f"Component {i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

# Plot 1: Centered data with principal components and transformed data (Y_tilde)
plt.figure(figsize=(10, 6))

# Get centered data
X_centered = data - pca_full.mean

# Plot centered data
plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.7, label='Centered data')

# Plot principal components (eigenvectors)
for i in range(pca_full.components.shape[1]):
    plt.arrow(0, 0,
              pca_full.components[0, i] * np.sqrt(pca_full.explained_variance[i]),
              pca_full.components[1, i] * np.sqrt(pca_full.explained_variance[i]),
              head_width=0.1, head_length=0.1, 
              fc=plt.cm.tab10(i), ec=plt.cm.tab10(i),
              label=f'Principal Component {i+1}')

# Plot transformed data in original space (Y_tilde)
plt.scatter(data_transformed_full[:, 0], np.zeros_like(data_transformed_full[:, 0]), 
            alpha=0.7, marker='x', s=30, c='red', label='Y_tilde (1st component)')
if data_transformed_full.shape[1] > 1:
    plt.scatter(np.zeros_like(data_transformed_full[:, 1]), data_transformed_full[:, 1], 
                alpha=0.7, marker='x', s=30, c='green', label='Y_tilde (2nd component)')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Centered Data with Principal Components and Y_tilde')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True)
plt.legend()
plt.savefig('./ML_ADVANCED/LAB4/LAB/plots/centered_data_with_pca.png')
plt.show()

# Plot 2: Original data with principal components and reconstructed data (Y_hat)
plt.figure(figsize=(10, 6))

# Original data
plt.scatter(data[:, 0], data[:, 1], alpha=0.7, label='Original data')

# Principal components around the mean
for i in range(pca_full.components.shape[1]):
    plt.arrow(pca_full.mean[0], pca_full.mean[1],
              pca_full.components[0, i] * np.sqrt(pca_full.explained_variance[i]) * 2,
              pca_full.components[1, i] * np.sqrt(pca_full.explained_variance[i]) * 2,
              head_width=0.1, head_length=0.1, 
              fc=plt.cm.tab10(i), ec=plt.cm.tab10(i),
              label=f'Principal Component {i+1}')

# Reconstructed data (Y_hat)
data_reconstructed_full = pca_full.inverse_transform(data_transformed_full)
plt.scatter(data_reconstructed_full[:, 0], data_reconstructed_full[:, 1], 
            alpha=0.5, marker='o', edgecolors='red', facecolors='none', 
            s=80, label='Y_hat (Reconstructed)')

# Draw lines connecting original and reconstructed points
for i in range(len(data)):
    plt.plot([data[i, 0], data_reconstructed_full[i, 0]], 
             [data[i, 1], data_reconstructed_full[i, 1]], 
             'k-', alpha=0.2)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original Data with Principal Components and Y_hat')
plt.grid(True)
plt.legend()
plt.savefig('./ML_ADVANCED/LAB4/LAB/plots/original_data_with_reconstructed.png')
plt.show()

