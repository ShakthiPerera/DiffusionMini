import numpy as np
from sklearn.preprocessing import MinMaxScaler

def banana_cen_pdf(theta, max_distance, concentration_factor=2, decay=0.5):
    r = max_distance * (np.exp(-decay * theta) * (1 + concentration_factor * np.sin(theta)))
    return r

def generate_cen_banana_points(num_points, sigma=0.3, max_distance=1.0):
    X = []
    np.random.seed(42)
    for _ in range(num_points):
        theta = np.random.exponential(0.25)
        r = banana_cen_pdf(theta, max_distance)
        width_effect = sigma * (1 - (r / max_distance))
        x_center = r * np.cos(theta)
        y_center = r * np.sin(theta)
        x_offset = np.random.uniform(-width_effect / 2, width_effect / 2)
        y_offset = np.random.uniform(-width_effect / 2, width_effect / 2)
        x_new = x_center + x_offset
        y_new = y_center + y_offset
        X.append((x_new, y_new))
    return np.array(X)

def generate_central_points(num_points, max_distance=0.5, sigma=0.1):
    central_points = []
    np.random.seed(42)
    for _ in range(num_points):
        theta = np.pi / 3 + np.random.normal(0, 0.5)
        r = max_distance * (np.random.uniform(0.5, 0.8))
        x_center = r * np.cos(theta)
        y_center = r * np.sin(theta)
        x_offset = np.random.normal(0, sigma / 2)
        y_offset = np.random.normal(0, sigma / 2)
        central_points.append((x_center + x_offset, y_center + y_offset))
    return np.array(central_points)

def generate_cen_banana(total_points=10000, central_points=500):
    banana_points = generate_cen_banana_points(total_points - central_points)
    middle_points = generate_central_points(central_points)
    all_points = np.vstack((banana_points, middle_points))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_rescaled = scaler.fit_transform(all_points)
    return x_rescaled