import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_moons

def create_crescent_with_scatter(num_samples=10000, noise_level=0.1, random_state=42,
                                            crescent_class=0, scatter_band=0.3, scatter_density=500):
    # Generate the two-class moons dataset and filter for one crescent
    X, y = make_moons(2*num_samples - 2*scatter_density, noise=noise_level, random_state=random_state)
    X_single_crescent = X[y == crescent_class]

    np.random.seed(random_state)

    X_scatter = []
    for point in X_single_crescent:
        x_crescent, y_crescent = point

        # Add noise to the crescent points within a specified scatter band
        x_scattered = x_crescent + scatter_band * np.random.randn()
        y_scattered = y_crescent + scatter_band * np.random.randn()
        X_scatter.append((x_scattered, y_scattered))

    X_scatter = np.array(X_scatter[:scatter_density])  # Limit the scatter points to the specified density

    X_combined = np.vstack((X_single_crescent, X_scatter))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_crescent_normalized = scaler.fit_transform(X_combined)

    return X_crescent_normalized