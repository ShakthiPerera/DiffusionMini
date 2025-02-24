import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_moons

def create_crescent_with_circles(num_samples=10000, noise_level=0.1, random_state=42,
                                            crescent_class=0, circle_center1=(-1.5, -1.0), circle_center2=(1.5, -1.0), circle_radius=0.2, circle_density=500):
    # Generate the two-class moons dataset and filter for one crescent
    X, y = make_moons(2*num_samples-4*circle_density, noise=noise_level, random_state=random_state)
    X_single_crescent = X[y == crescent_class]

    np.random.seed(random_state)

    angles1 = 2 * np.pi * np.random.rand(circle_density)  # Random angles for circular distribution
    angles2 = 2 * np.pi * np.random.rand(circle_density)
    radii1 = circle_radius * np.sqrt(np.random.rand(circle_density))  # Scaled radii for dense clustering
    radii2 = circle_radius * np.sqrt(np.random.rand(circle_density))

    radii1 += 0.05 * np.random.randn(circle_density)
    radii2 += 0.05 * np.random.randn(circle_density)
    # Convert polar coordinates to Cartesian for the small circle points
    X_circle1 = np.c_[radii1 * np.cos(angles1) + circle_center1[0], radii1 * np.sin(angles1) + circle_center1[1]]
    X_circle2 = np.c_[radii2 * np.cos(angles2) + circle_center2[0], radii2 * np.sin(angles2) + circle_center2[1]]


    # Combine crescent and small circular cluster points
    X_combined = np.vstack((X_single_crescent, X_circle1, X_circle2))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_crescent_normalized = scaler.fit_transform(X_combined)

    return X_crescent_normalized