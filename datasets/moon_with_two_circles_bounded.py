import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_moons

def create_bounded_crescent_with_circles(
        num_samples=10000, noise_level=0.1, random_state=42,
        crescent_class=0, crescent_bounds=(-2, 2, -1.5, 1.5),
        circle_center1=(-1.5, -1.0), circle_center2=(1.5, -1.0),
        circle_inner_radius=0.01, circle_outer_radius=0.25, circle_density=500):

    np.random.seed(random_state)

    # Generate 20% more data initially
    initial_samples = int(num_samples * 1.2)
    X, y = make_moons(n_samples=2 * initial_samples, noise=noise_level, random_state=random_state)
    X_single_crescent = X[y == crescent_class]

    # Generate points for two small circles within the given radii constraints
    def generate_circle_points(center, inner_radius, outer_radius, num_points):
        angles = 2 * np.pi * np.random.rand(num_points)
        radii = np.sqrt(np.random.uniform(inner_radius**2, outer_radius**2, num_points))
        X_circle = np.c_[radii * np.cos(angles) + center[0], radii * np.sin(angles) + center[1]]
        return X_circle

    X_circle1 = generate_circle_points(circle_center1, circle_inner_radius, circle_outer_radius, circle_density)
    X_circle2 = generate_circle_points(circle_center2, circle_inner_radius, circle_outer_radius, circle_density)

    # Combine crescent and circular cluster points
    X_combined = np.vstack((X_single_crescent, X_circle1, X_circle2))

    # Apply a single mask for bounding conditions
    mask = (
        (X_combined[:, 0] >= crescent_bounds[0]) & (X_combined[:, 0] <= crescent_bounds[1]) &
        (X_combined[:, 1] >= crescent_bounds[2]) & (X_combined[:, 1] <= crescent_bounds[3])
    )
    X_filtered = X_combined[mask]

    # Shuffle and select exactly `num_samples` points
    np.random.shuffle(X_filtered)
    X_final = X_filtered[:num_samples]

    # Normalize dataset to range (-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_normalized = scaler.fit_transform(X_final)

    return X_normalized