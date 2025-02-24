import numpy as np
from sklearn.preprocessing import MinMaxScaler

def generate_ring_points(center, inner_radius, outer_radius, num_points):
    angles = 2 * np.pi * np.random.rand(num_points)  # Random angles
    radii = np.sqrt(np.random.uniform(inner_radius**2, outer_radius**2, num_points))  # Random radii between inner and outer circles
    X_ring = np.c_[radii * np.cos(angles) + center[0], radii * np.sin(angles) + center[1]]  # Cartesian coordinates
    return X_ring

def create_two_ring_dataset(num_samples=10000, random_state=42,
                            center=(0, 0), inner_circle_radius1=0.1, outer_circle_radius1=0.2, inner_circle_radius2=0.3, outer_circle_radius2=0.4):
    np.random.seed(random_state)

    # Generate points for the first ring (using inner and outer radii)
    X_ring1 = generate_ring_points(center, inner_circle_radius1, outer_circle_radius1, num_samples//2)

    # Generate points for the second ring (with the same center and radii)
    X_ring2 = generate_ring_points(center, inner_circle_radius2, outer_circle_radius2, num_samples//2)

    # Combine the two sets of points to form two distinct rings
    X_combined = np.vstack((X_ring1, X_ring2))

    # Normalize the dataset to the range (-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_combined_normalized = scaler.fit_transform(X_combined)

    return X_combined_normalized