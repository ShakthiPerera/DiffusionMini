import numpy as np
from sklearn.preprocessing import MinMaxScaler

def banana_pdf(theta, max_distance, concentration_factor=2, decay=0.5):
    return max_distance * (np.exp(-decay * theta) * (1 + concentration_factor * np.sin(theta)))

def generate_banana_points(num_points=5000, sigma=0.3, max_distance=1.0):
    X = []
    for _ in range(num_points):
        theta = np.random.exponential(0.2)
        r = banana_pdf(theta, max_distance)
        width_effect = sigma * (1 - (r / max_distance))

        x_center = r * np.cos(theta)
        y_center = r * np.sin(theta)

        x_offset = np.random.uniform(-width_effect / 2, width_effect / 2)
        y_offset = np.random.uniform(-width_effect / 2, width_effect / 2)
        X.append((x_center + x_offset, y_center + y_offset))

    return np.array(X)

def create_banana_with_noisy_circles_and_local_scatter(num_points=10000, circle_center1=(-0.4, 0.5),
                                                       circle_center2=(0.5, -0.25), circle_radius=0.075,
                                                       circle_density=500, circle_noise=0.05, scatter_density=500,
                                                       noise_band=0.1, random_state=42):
    np.random.seed(random_state)

    X_banana = generate_banana_points(num_points - 2 * circle_density - scatter_density)

    angles1 = 2 * np.pi * np.random.rand(circle_density)
    angles2 = 2 * np.pi * np.random.rand(circle_density)
    radii1 = circle_radius * np.sqrt(np.random.rand(circle_density)) + circle_noise * np.random.randn(circle_density)
    radii2 = circle_radius * np.sqrt(np.random.rand(circle_density)) + circle_noise * np.random.randn(circle_density)

    X_circle1 = np.c_[radii1 * np.cos(angles1) + circle_center1[0], radii1 * np.sin(angles1) + circle_center1[1]]
    X_circle2 = np.c_[radii2 * np.cos(angles2) + circle_center2[0], radii2 * np.sin(angles2) + circle_center2[1]]

    X_scatter = []
    for _ in range(scatter_density):
        theta = np.random.exponential(0.2)
        r = banana_pdf(theta, max_distance=1.0) * (1 + noise_band * np.random.randn())

        x_noisy = r * np.cos(theta)
        y_noisy = r * np.sin(theta)
        X_scatter.append((x_noisy, y_noisy))

    X_scatter = np.array(X_scatter)

    X_combined = np.vstack((X_banana, X_circle1, X_circle2, X_scatter))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_normalized = scaler.fit_transform(X_combined)

    return X_normalized