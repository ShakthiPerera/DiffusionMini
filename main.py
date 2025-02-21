from runners.ddpm_default import DDPM as ddpm
from models import ConditionalDenseModel
from functions import make_beta_schedule
import torch 
from tqdm import tqdm 
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import MinMaxScaler

X, _ = make_swiss_roll(10000, noise=0.5, random_state=42)

# Restrict to 2D
X = X[:,[0,2]]

# Normalize to be between -1 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))
X_normalized = scaler.fit_transform(X)

X_train, X_val = train_test_split(X_normalized, test_size=0.2)

X_train = torch.tensor(X_train).float()
X_val = torch.tensor(X_val).float()

train_set = TensorDataset(X_train)
val_set = TensorDataset(X_val)

batch_size = 64

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    drop_last=True,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    drop_last=False,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


def create_model(num_features=[2, 128, 128, 128, 2]):

    eps_model = ConditionalDenseModel(num_features, activation='relu', embed_dim=10)
    betas = make_beta_schedule(num_steps=1000, mode='linear', beta_range=(1e-04, 0.02))

    ddpm_model = ddpm(eps_model=eps_model, betas=betas, criterion='mse', lr=1e-03)

    return ddpm_model

def train(ddpm, device, num_epochs=1000):
    # Training loop

    train_losses = torch.zeros(num_epochs)
    diff_losses = torch.zeros(num_epochs)
    norm_losses = torch.zeros(num_epochs)

    ddpm.to(device)
    training_length = len(train_loader)

    for epoch in range(num_epochs):
        ddpm.train()
        loss, simple_diff_loss, norms = 0, 0, 0
        for x_batch in tqdm(train_loader):
            x_batch = torch.stack(x_batch).to(device)
            loss_, simple_loss, norm_loss = ddpm.train_step(x_batch)
            loss += loss_
            simple_diff_loss += simple_loss
            norms += norm_loss

        loss = loss / training_length
        avg_diff_loss = simple_diff_loss / training_length
        avg_norm_loss = norms / training_length

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss:.4f}')

        # Save losses
        train_losses[epoch] = loss
        diff_losses[epoch] = avg_diff_loss
        norm_losses[epoch] = avg_norm_loss

    return train_losses, diff_losses, norm_losses

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    print("Started")
    ddpm_model = create_model()
    print("Model Created")
    train_losses, diff_losses, norm_losses = train(ddpm_model, device=device)