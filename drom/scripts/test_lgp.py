import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse

from drom.models.generic import LatentGoalPredictor
from drom.torch_utils.torch_utils import freeze_torch_model_params


def load_model(checkpoint_path, state_dim, hidden_dim=256, goal_dim=7, device='cpu'):
    model = LatentGoalPredictor(state_dim=state_dim, hidden_dim=hidden_dim, goal_dim=goal_dim)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, test_loader, num_episodes, device):
    loss_fn = torch.nn.MSELoss()
    total_loss = 0.0
    total_samples = 0

    all_loss = []
    all_preds = []
    all_targets = []

    ep_counter = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            all_loss.append(loss.item())
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
            if ep_counter == num_episodes - 1:
                break

            ep_counter += 1

    avg_loss = total_loss / total_samples
    # preds = torch.cat(all_preds, dim=0).numpy()
    # targets = torch.cat(all_targets, dim=0).numpy()

    return avg_loss, all_loss, all_preds, all_targets


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Using device: {device}")

    # Compose dataset path
    base_path = "lgp_models"
    path = os.path.join(base_path, args.directory)
    dir_path = os.path.join(path, args.run_name)
    dataset_path = os.path.join(dir_path, "val_dataset.pt")

    # Load dataset
    test_dataset = torch.load(dataset_path, weights_only=False)  # assuming it returns a dict with 'states' and 'actions'
    # Create DataLoader
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Load model (assuming load_model is defined elsewhere)
    model_path = os.path.join(dir_path, f"checkpoints/model_epoch_{args.checkpoint}.pth")
    model = load_model(
        checkpoint_path=model_path,
        state_dim=test_dataset[0][0].shape[0],
        hidden_dim=args.hidden_dim,
        goal_dim=test_dataset[0][1].shape[0],
        device=device,
    )

    freeze_torch_model_params(model)  # assuming this freezes the parameters

    # Compile model if using PyTorch 2.0+
    model = torch.compile(model)

    # Warmup model if warmup method exists
    if hasattr(model, "warmup"):
        model.warmup(device=device)

    # Evaluate model (assuming evaluate_model is defined elsewhere)
    test_loss, losses, preds, targets = evaluate_model(model, test_dataset, args.episodes, device)

    # Print examples if requested
    if args.print_examples:
        for i in range(len(preds)):
            print(f"\nExample {i + 1}:")
            print(f"  Target goal:    {targets[i].tolist()}")
            print(f"  Predicted goal: {preds[i].tolist()}")
            print(f"  Sample loss: {losses[i]}")
    
    print(f"\n\n✅ Test Loss (MSE): {test_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True, help="Directory storing torch trained models.")
    parser.add_argument("--run_name", type=str, required=True, help="Directory storing torch trained models.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to be loaded.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of validation trials.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--print-examples", action="store_true")
    args = parser.parse_args()

    main(args)
