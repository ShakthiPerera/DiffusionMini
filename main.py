import argparse 
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader

from datasets import EightGaussiansDataset, MoonDataset, BananaWithTwoCirclesDataset, SCurveDataset, SwissRollDataset
from models import ConditionalDenseModel
from functions import make_beta_schedule
from runners.ddpm_default import DDPM as ddpm

def parse_args():
    def restricted_float(x):
        """Ensure reg is between 0 and 1, excluding 0."""
        x = float(x)
        if x <= 0 or x > 1:
            raise argparse.ArgumentTypeError(f"reg must be in (0, 1], got {x}")
        return x

    def positive_int(x):
        """Ensure the integer is greater than 0."""
        x = int(x)
        if x <= 0:
            raise argparse.ArgumentTypeError(f"{x} must be greater than 0")
        return x

    def non_negative_int(x):
        """Ensure the integer is non-negative (for gpu_id)."""
        x = int(x)
        if x < 0:
            raise argparse.ArgumentTypeError(f"{x} must be non-negative")
        return x

    def valid_loss_type(x):
        """Restrict loss_type to 'default' or 'iso'."""
        x = x.lower()
        if x not in ['default', 'iso']:
            raise argparse.ArgumentTypeError(f"loss_type must be 'default' or 'iso', got {x}")
        return x

    parser = argparse.ArgumentParser(description="Model training script")

    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (required)')
    parser.add_argument('--loss_type', type=valid_loss_type, default='default',
                        help='Loss function type: "default" or "iso" (default: default)')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Directory to save logs (required)')
    parser.add_argument('--reg', type=restricted_float, default=0.0,
                        help='Regularization strength in (0, 1] (default: 0.0)')
    parser.add_argument('--num_epoch', type=positive_int, default=1000,
                        help='Number of training epochs, > 0 (default: 1000)')
    parser.add_argument('--gpu_id', type=non_negative_int, default=0,
                        help='GPU device ID, single non-negative integer (default: 0)')
    parser.add_argument('--num_samples', type=positive_int, default=10000,
                        help='Number of samples for dataset creation/sampling, > 0 (default: 10000)')
    parser.add_argument('--seed', type=positive_int, default=42,
                        help='Random state number for reproducibility')
    parser.add_argument('-b', '--batch_size', type=positive_int, default=64,
                        help='Size of the batches for training.')
    parser.add_argument('--schedule', type=str, default='linear',
                        help='beta schedule type')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--hl', type=list, default=[2, 128, 128, 128, 2],
                        help='hidden layers size for noise prediction')
    args = parser.parse_args()
    return args

def load_dataset(dataset_name, num_samples, batch_size, random_state):
    '''
    Loading the dataset for given argeparse properties

    Args:
        dataset_name (str): name of the dataset 
        num_samples (int): number of samples you want
        batch_size (int) : batch size for the dataset 
        random_state (int): random state fro reproducibility
    '''
    if dataset_name == "8-gaussian":
        ds = EightGaussiansDataset(num_samples, random_state)
        X = ds.generate()
    elif dataset_name == "swiss_roll":
        ds = SwissRollDataset(num_samples, random_state)
        X = ds.generate()
    elif dataset_name == "moons":
        ds = MoonDataset(num_samples, random_state)
        X = ds.generate()
    elif dataset_name == "s_curve":
        ds = SCurveDataset(num_samples, random_state)
        X = ds.generate()
    elif dataset_name == "banana_with_two_circles":
        ds = BananaWithTwoCirclesDataset(num_samples, random_state)
        X = ds.generate()

    X_train, _ = train_test_split(X, test_size=0.2)
    X_train = torch.tensor(X_train).float()
    train_set = TensorDataset(X_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=2, pin_memory=True)

    return ds, train_loader

def create_model(loss_type, schedule_type, learning_rate, num_features):
    '''
    Creating the DDPM model instance for the training

    Args:
        loss_type (str): loss type (iso, default)
        schedule_type (str): schedule variation type (linear, quadratic, cosine ..)
        learning_rate (float): models' learning rate
        num_features (list, optional): Conditional models hidden layers sizes for noise prediction. Defaults to [2, 128, 128, 128, 2].

    Returns:
        torch.model: model instanace
    '''
    eps_model = ConditionalDenseModel(num_features, activation='relu', embed_dim=10)
    betas = make_beta_schedule(num_steps=1000, mode=schedule_type, beta_range=(1e-04, 0.02))

    ddpm_model = ddpm(eps_model=eps_model, betas=betas, criterion='mse', lr=learning_rate, loss_type=loss_type)

    return ddpm_model

if __name__ == "__main__":
    args = parse_args()
    print(f"Dataset: {args.dataset}")
    print(f"Loss Type: {args.loss_type}")
    print(f"Log Dir: {args.log_dir}")
    print(f"Reg: {args.reg}")
    print(f"Num Epochs: {args.num_epoch}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Num Samples: {args.num_samples}")
    print(f"Random State: {args.seed}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Schedule Type: {args.schedule}")
    print(f"Learning Rate: {args.lr}")
    print(f"Hidden Layer Dims: {args.hl}")

    ds, X = load_dataset(args.dataset, args.num_samples, args.batch_size, args.seed)
    print(type(X))
    ds.plot_dataset()
    model = create_model(args.loss_type, args.schedule, args.lr, args.hl)
    print(model)



# def train(ddpm, device, num_epochs=1000):
#     # Training loop

#     train_losses = torch.zeros(num_epochs)
#     diff_losses = torch.zeros(num_epochs)
#     norm_losses = torch.zeros(num_epochs)

#     ddpm.to(device)
#     training_length = len(train_loader)

#     for epoch in range(num_epochs):
#         ddpm.train()
#         loss, simple_diff_loss, norms = 0, 0, 0
#         for x_batch in tqdm(train_loader):
#             x_batch = torch.stack(x_batch).to(device)
#             loss_, simple_loss, norm_loss = ddpm.train_step(x_batch)
#             loss += loss_
#             simple_diff_loss += simple_loss
#             norms += norm_loss

#         loss = loss / training_length
#         avg_diff_loss = simple_diff_loss / training_length
#         avg_norm_loss = norms / training_length

#         print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss:.4f}')

#         # Save losses
#         train_losses[epoch] = loss
#         diff_losses[epoch] = avg_diff_loss
#         norm_losses[epoch] = avg_norm_loss

#     return train_losses, diff_losses, norm_losses

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# if __name__ == "__main__":
#     print("Started")
#     ddpm_model = create_model()
#     print("Model Created")
#     train_losses, diff_losses, norm_losses = train(ddpm_model, device=device)