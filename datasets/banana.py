import numpy as np
from sklearn.preprocessing import MinMaxScaler

def banana_pdf(theta, max_distance, concentration_factor=2, decay=0.5):
    """
    Define the radius for banana shape with density concentrated on one end.
    """
    # Make radius dependent on angle to create an elongated "banana" effect
    r = max_distance * (np.exp(-decay * theta) * (1 + concentration_factor * np.sin(theta)))
    return r

def generate_banana_points(num_points=5000, sigma=0.3, max_distance=1.0):
    """
    Generate points according to the defined probability distribution for a banana shape.
    """
    X = []
    np.random.seed(42)
    for _ in range(num_points):
        theta = theta = np.random.exponential(0.25) #np.random.uniform(0, np.pi/2)  # Random angle for half-circle (banana shape)
        r = banana_pdf(theta, max_distance)

        # Calculate width effect based on distance
        width_effect = sigma * (1 - (r / max_distance))  # Linearly decrease width with distance

        # Convert to Cartesian coordinates
        x_center = r * np.cos(theta)
        y_center = r * np.sin(theta)

        # Generate random offsets based on the current width
        x_offset = np.random.uniform(-width_effect / 2, width_effect / 2)
        y_offset = np.random.uniform(-width_effect / 2, width_effect / 2)

        # Add offsets to the center point
        x_new = x_center + x_offset
        y_new = y_center + y_offset

        # Add to the lists
        X.append((x_new, y_new))

    return np.array(X)

def generate_banana(num_points=10000):
    x = generate_banana_points(num_points)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_rescaled = scaler.fit_transform(x)
    return x_rescaled