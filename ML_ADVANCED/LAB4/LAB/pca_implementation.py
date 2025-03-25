import matplotlib.pyplot as plt
import numpy as np


# Function to plot centered data with principal components and Y_tilde
def plot_centered_data_with_pca(X, pca, X_transformed):
    """
    Plot the centered training data with principal components and Y_tilde.
    
    Args:
        X: Original data
        pca: Fitted PCA object
        X_transformed: Transformed data (Y_tilde)
    """
    plt.figure(figsize=(10, 6))

    # Get centered data
    X_centered = X - pca.mean

    # Plot centered data
    plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.7, label='Centered data')

    # Plot principal components (eigenvectors)
    for i in range(pca.components.shape[1]):
        plt.arrow(0, 0,
                  pca.components[0, i] * np.sqrt(pca.explained_variance[i]),
                  pca.components[1, i] * np.sqrt(pca.explained_variance[i]),
                  head_width=0.1, head_length=0.1,
                  fc=plt.cm.tab10(i), ec=plt.cm.tab10(i),
                  label=f'Principal Component {i + 1}')

    # Plot transformed data in original space (Y_tilde)
    plt.scatter(X_transformed[:, 0], np.zeros_like(X_transformed[:, 0]),
                alpha=0.7, marker='x', s=30, c='red', label='Y_tilde (1st component)')
    if X_transformed.shape[1] > 1:
        plt.scatter(np.zeros_like(X_transformed[:, 1]), X_transformed[:, 1],
                    alpha=0.7, marker='x', s=30, c='green', label='Y_tilde (2nd component)')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Centered Data with Principal Components and Y_tilde')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True)
    plt.legend()
    plt.show()


# Function to plot original data with principal components and Y_hat
def plot_original_data_with_reconstruction(X, pca, X_transformed):
    """
    Plot the original data with principal components and reconstructed data (Y_hat).
    
    Args:
        X: Original data
        pca: Fitted PCA object
        X_transformed: Transformed data (Y_tilde)
    """
    plt.figure(figsize=(10, 6))
    
    # Original data
    plt.scatter(X[:, 0], X[:, 1], alpha=0.7, label='Original data')
    
    # Principal components around the mean
    for i in range(pca.components.shape[1]):
        plt.arrow(pca.mean[0], pca.mean[1],
                  pca.components[0, i] * np.sqrt(pca.explained_variance[i]) * 2,
                  pca.components[1, i] * np.sqrt(pca.explained_variance[i]) * 2,
                  head_width=0.1, head_length=0.1,
                  fc=plt.cm.tab10(i), ec=plt.cm.tab10(i),
                  label=f'Principal Component {i + 1}')
    
    # Reconstructed data (Y_hat)
    Y_hat = pca.inverse_transform(X_transformed)
    plt.scatter(Y_hat[:, 0], Y_hat[:, 1],
                alpha=0.5, marker='o', edgecolors='red', facecolors='none',
                s=80, label='Y_hat (Reconstructed)')
    
    # Draw lines connecting original and reconstructed points
    for i in range(len(X)):
        plt.plot([X[i, 0], Y_hat[i, 0]],
                 [X[i, 1], Y_hat[i, 1]],
                 'k-', alpha=0.2)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Original Data with Principal Components and Y_hat')
    plt.grid(True)
    plt.legend()
    plt.show()


# Function to plot test results in both centered and original spaces
def plot_test_results(test_data, pca, test_transformed):
    """
    Plot test results on N-dimensional space for both the centered and original data.
    
    Args:
        test_data: Test data
        pca: Fitted PCA object
        test_transformed: Transformed test data
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot 1: Centered test data
    # Get centered test data
    test_centered = test_data - pca.mean
    
    # Plot centered test data
    axes[0].scatter(test_centered[:, 0], test_centered[:, 1], alpha=0.7, color='purple', 
                   label='Centered test data')
    
    # Plot principal components (eigenvectors)
    for i in range(pca.components.shape[1]):
        axes[0].arrow(0, 0,
                    pca.components[0, i] * np.sqrt(pca.explained_variance[i]),
                    pca.components[1, i] * np.sqrt(pca.explained_variance[i]),
                    head_width=0.1, head_length=0.1, 
                    fc=plt.cm.tab10(i), ec=plt.cm.tab10(i),
                    label=f'Principal Component {i + 1}')
    
    # Plot transformed test data projections
    axes[0].scatter(test_transformed[:, 0], np.zeros_like(test_transformed[:, 0]), 
                   alpha=0.7, marker='x', s=40, color='red', 
                   label='Test Y_tilde (1st component)')
    if test_transformed.shape[1] > 1:
        axes[0].scatter(np.zeros_like(test_transformed[:, 1]), test_transformed[:, 1], 
                        alpha=0.7, marker='x', s=40, color='green', 
                        label='Test Y_tilde (2nd component)')
    
    # Plot connections between centered data and projections
    for i in range(len(test_centered)):
        axes[0].plot([test_centered[i, 0], test_transformed[i, 0]], 
                    [test_centered[i, 1], 0], 
                    'r--', alpha=0.3)
        if test_transformed.shape[1] > 1:
            axes[0].plot([test_centered[i, 0], 0], 
                         [test_centered[i, 1], test_transformed[i, 1]], 
                         'g--', alpha=0.3)
    
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].set_title('Centered Test Data with Principal Components')
    axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot 2: Original test data
    # Original test data
    axes[1].scatter(test_data[:, 0], test_data[:, 1], alpha=0.7, color='purple', 
                    label='Original test data')
    
    # Principal components around the mean
    for i in range(pca.components.shape[1]):
        axes[1].arrow(pca.mean[0], pca.mean[1],
                      pca.components[0, i] * np.sqrt(pca.explained_variance[i]) * 2,
                      pca.components[1, i] * np.sqrt(pca.explained_variance[i]) * 2,
                      head_width=0.1, head_length=0.1, 
                      fc=plt.cm.tab10(i), ec=plt.cm.tab10(i),
                      label=f'Principal Component {i + 1}')
    
    # Reconstructed test data
    test_reconstructed = pca.inverse_transform(test_transformed)
    axes[1].scatter(test_reconstructed[:, 0], test_reconstructed[:, 1], 
                    alpha=0.5, marker='o', edgecolors='red', facecolors='none', 
                    s=80, label='Y_tilde (Reconstructed test)')
    
    # Draw lines connecting original and reconstructed test points
    for i in range(len(test_data)):
        axes[1].plot([test_data[i, 0], test_reconstructed[i, 0]], 
                     [test_data[i, 1], test_reconstructed[i, 1]], 
                     'k-', alpha=0.3)
    
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].set_title('Original Test Data with Principal Components and Y_tilde')
    axes[1].grid(True)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


# Load the data
data = np.loadtxt('./ML_ADVANCED/LAB4/LAB/data/data_pca.txt')
print(f"Data shape: {data.shape}")

# Print the first 5 rows of the data
print(f"First 5 rows of the data: \n {data[:5]}")


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
pca = PCA(n_components=n_components)
pca.fit(data)

# Get the transformed data
data_transformed_full = pca.transform(data)
print(f"Transformed data shape: {data_transformed_full.shape}")
print(f"Transformed data from the first 5 rows: \n {data_transformed_full[:5]}")

# Calculate explained variance ratio
explained_var_ratio = pca.explained_variance / np.sum(pca.explained_variance)
print("Explained variance ratio by component:")
for i, ratio in enumerate(explained_var_ratio):
    print(f"Component {i + 1}: {ratio:.4f} ({ratio * 100:.2f}%)")

# Load test data and apply the PCA model
print("\n--- Testing with test.txt data ---")
test_data = np.loadtxt('./ML_ADVANCED/LAB4/LAB/data/test.txt')
print(f"Test data shape: {test_data.shape}")

# Transform test data
test_transformed = pca.transform(test_data)
test_reconstructed = pca.inverse_transform(test_transformed)

print("\n--- Results for test.txt data ---")
print(f"Original test data shape: {test_data.shape}")
print(f"Transformed test data shape: {test_transformed.shape}")
print(f"Reconstructed test data shape: {test_reconstructed.shape}")

# Plot centered training data with principal components and Y_tilde
plot_centered_data_with_pca(data, pca, data_transformed_full)

# Plot original data with principal components and reconstructed data
plot_original_data_with_reconstruction(data, pca, data_transformed_full)

# Plot test results in both centered and original spaces
plot_test_results(test_data, pca, test_transformed)
