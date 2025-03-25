import numpy as np

# Load the data
data = np.loadtxt('./ML_ADVANCED/LAB4/LAB/data/data_pca.txt')
print(f"Data shape: {data.shape}")


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
        X_transformed = np.dot(X_centered, self.components)
        return X_transformed

    def inverse_transform(self, X_transformed):
        # 7. Reconstruct the data in the original space
        X_reconstructed = np.dot(X_transformed, self.components.T) + self.mean
        return X_reconstructed


n_components = data.shape[1]

# Create and fit PCA with full components
pca_full = PCA(n_components=n_components)
pca_full.fit(data)

# Get the transformed data
data_transformed_full = pca_full.transform(data)
print(f"Transformed data shape: {data_transformed_full.shape}")

# Calculate explained variance ratio
explained_var_ratio = pca_full.explained_variance / np.sum(pca_full.explained_variance)
print("Explained variance ratio by component:")
for i, ratio in enumerate(explained_var_ratio):
    print(f"Component {i + 1}: {ratio:.4f} ({ratio * 100:.2f}%)")

# Load test data and apply the PCA model
print("\n--- Testing with test.txt data ---")
test_data = np.loadtxt('./ML_ADVANCED/LAB4/LAB/data/test.txt')
print(f"Test data shape: {test_data.shape}")

# Transform test data
test_transformed = pca_full.transform(test_data)
test_reconstructed = pca_full.inverse_transform(test_transformed)

print("\n--- Results for test.txt data ---")
print(f"Original test data shape: {test_data.shape}")
print(f"Transformed test data shape: {test_transformed.shape}")
print(f"Reconstructed test data shape: {test_reconstructed.shape}")


