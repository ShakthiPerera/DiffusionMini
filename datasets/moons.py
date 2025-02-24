from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler

def create_moons_samples(num_samples=10000, noise_level=0.1, random_state = 42):

  X, y = make_moons(num_samples, noise=noise_level, random_state=random_state)

  # Normalize to be between -1 and 1
  scaler = MinMaxScaler(feature_range=(-1, 1))
  X_normalized = scaler.fit_transform(X)

  return X_normalized, y