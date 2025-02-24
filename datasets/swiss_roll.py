from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import MinMaxScaler

def create_swiss_roll_samples(num_samples=10000, noise_level=0.5, random_state=42):
  X, _ = make_swiss_roll(num_samples, noise=noise_level, random_state=random_state)

  # Restrict to 2D
  X = X[:,[0,2]]

  # Normalize to be between -1 and 1
  scaler = MinMaxScaler(feature_range=(-1, 1))
  X_normalized = scaler.fit_transform(X)

  return X_normalized