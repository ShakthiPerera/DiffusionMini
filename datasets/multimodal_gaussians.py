import numpy as np
from sklearn.preprocessing import MinMaxScaler

def generate_CombinedGaussian_distributions(num_points=10000):
    """
    Generate points from Gaussian, Sub-Gaussian, and Super-Gaussian distributions within -1 and 1 range.
    """
    X = []
    # Gaussian (centered at origin, narrow spread)
    mean_gaussian_x = 0
    mean_gaussian_y = 0
    std_gaussian_x = 0.075
    std_gaussian_y = 0.075

    # Sub-Gaussian (slightly shifted to the left, moderate spread)
    mean_subgaussian_x = -0.25
    mean_subgaussian_y = 0.25
    std_subgaussian_x = 0.1
    std_subgaussian_y = 0.1

    # Super-Gaussian (slightly shifted to the right, wider spread)
    mean_supergaussian_x = 0.25
    mean_supergaussian_y = -0.25
    std_supergaussian_x = 0.15
    std_supergaussian_y = 0.15

    # Generate points for each distribution
    x_gaussian = np.random.normal(loc=mean_gaussian_x, scale=std_gaussian_x, size=num_points//3)
    y_gaussian = np.random.normal(loc=mean_gaussian_y, scale=std_gaussian_y, size=num_points//3)

    x_subgaussian = np.random.normal(loc=mean_subgaussian_x, scale=std_subgaussian_x, size=num_points//3)
    y_subgaussian = np.random.normal(loc=mean_subgaussian_y, scale=std_subgaussian_y, size=num_points//3)

    x_supergaussian = np.random.normal(loc=mean_supergaussian_x, scale=std_supergaussian_x, size=num_points-2*num_points//3)
    y_supergaussian = np.random.normal(loc=mean_supergaussian_y, scale=std_supergaussian_y, size=num_points-2*num_points//3)

    # Combine all points
    x_combined1 = np.concatenate([x_gaussian, x_subgaussian, x_supergaussian])
    y_combined1 = np.concatenate([y_gaussian, y_subgaussian, y_supergaussian])

    # Rescale points to the range of -1 to +1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    combined_scaled = scaler.fit_transform(np.vstack((x_combined1, y_combined1)).T)

    # Separate x and y coordinates after scaling
    x_combined, y_combined = combined_scaled[:, 0], combined_scaled[:, 1]

    X.append((x_combined, y_combined))
    return np.array(X).reshape(-1, 2)