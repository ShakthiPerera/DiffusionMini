import numpy as np
from base_dataset_class import BaseDataset2D

class EightGaussiansDataset(BaseDataset2D):
    def __init__(self, num_samples=10000, dataset_name ='8_Gaussians', random_state=42):
        super().__init__(num_samples, dataset_name, random_state)
    
    def pdf(self, mean, covariance, size):
        rng = np.random.default_rng(self.random_state)
        return rng.multivariate_normal(mean, covariance, size)
    
    def generate(self):
        # Number of data points per Gaussian
        n_points = int(self.num_samples / 8)

        radius = 8

        # Parameters for the Gaussian distributions
        means = [(radius/np.sqrt(2), radius/np.sqrt(2)), (radius/np.sqrt(2), -radius/np.sqrt(2)), (-radius/np.sqrt(2), radius/np.sqrt(2)), (-radius/np.sqrt(2), -radius/np.sqrt(2)),
                (0, -radius), (0, radius), (radius, 0), (-radius, 0)]
        covariance = np.eye(2)  # Identity matrix as covariance for all Gaussians

        # Generate data points for each Gaussian
        X = np.zeros((len(means) * n_points, 2))
        for i, mean in enumerate(means):
            X[i * n_points:(i + 1) * n_points] = self.pdf(mean, covariance, n_points)

        X = self.normalize(X)
        self.data = X
        return X