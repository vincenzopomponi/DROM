import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

import os
import h5py
import argparse
import numpy as np
import wandb
import time
from datetime import datetime

from drom.torch_utils.seed import fix_random_seed
from drom.models.generic import LatentGoalPredictor

# from robosuite.utils.transform_utils import quat2axisangle

# ------------------------------ #
#    Load RoboMimic Observations #
# ------------------------------ #
def extract_observations_from_robomimic(dataset_path, obs_keys, sequence_length, demo_limit=200):
    result = {key: [] for key in obs_keys}
    result["actions"] = []

    with h5py.File(dataset_path, 'r') as f:
        demos = list(f['data'].keys())[:demo_limit]
        print(f"Found {len(demos)} demos")

        for demo_id in demos:
            demo_data = f['data'][demo_id]
            demo_obs = demo_data['obs']

            # Replicate the first action
            first_action = demo_data['actions'][0]
            repeated_actions = np.repeat(first_action[np.newaxis, :], sequence_length, axis=0)
            actions = np.vstack([repeated_actions, demo_data['actions']])
            result["actions"].append(actions)

            # Replicate the first observation for each key
            for key in obs_keys:
                first_obs = demo_obs[key][0]
                repeated_obs = np.repeat(first_obs[np.newaxis, :], sequence_length, axis=0)
                obs = np.vstack([repeated_obs, demo_obs[key]])
                result[key].append(obs)

    # Concatenate all demos along the time dimension
    for key in result:
        result[key] = np.concatenate(result[key], axis=0)
        # input(f"{key}: {result[key].shape}")

    return result


# ------------------------------ #
#      Windowing for Training    #
# ------------------------------ #
def create_goal_prediction_windows(data, sequence_length, horizon):
    states = data["states"]
    actions = data["actions"]
    N = states.shape[0]

    max_start = N - sequence_length - horizon
    current_states = []
    goal_targets = []

    for t in range(max_start):
        window = states[t : t + sequence_length]
        future_goal = actions[t + sequence_length - 1 + horizon]
        current_states.append(window)  # Use last state in window
        goal_targets.append(future_goal)

    current_states = torch.tensor(np.array(current_states), dtype=torch.float32)
    goal_targets = torch.tensor(np.array(goal_targets), dtype=torch.float32)
    current_states = current_states.reshape(current_states.size(0), current_states.size(1)*current_states.size(2))
    # print(f"current_states: {current_states.shape}")
    # input(f"goal_targets: {goal_targets.shape}")

    return current_states, goal_targets


# ------------------------------ #
#         Training Loop          #
# ------------------------------ #
def train_latent_goal_predictor(
    states,
    target_goals,
    state_dim,
    hidden_dim=256,
    goal_dim=7,
    batch_size=128,
    num_epochs=100,
    lr=1e-3,
    val_split=0.1,
    device=None,
    val_every=50,
    save_dir=None,
    ckpt_dir=None,
    save_every=50,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LatentGoalPredictor(state_dim=state_dim, hidden_dim=hidden_dim, goal_dim=goal_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(states, target_goals)
    # torch.save(dataset, os.path.join(save_dir, f'dataset.pt'))
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # save the indices of training and validation sets (for later evaluation)
    torch.save(train_dataset, os.path.join(save_dir, f'train_dataset.pt'))
    torch.save(val_dataset, os.path.join(save_dir, f'val_dataset.pt'))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)


    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= train_size
        wandb.log({"train_loss": train_loss}, step=epoch)

        if (epoch + 1) % val_every == 0 or (epoch + 1) == num_epochs:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    val_loss += loss_fn(pred, y).item() * x.size(0)

            val_loss /= val_size
            wandb.log({"val_loss": val_loss}, step=epoch)
            print(f"[Epoch {epoch+1:03d}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        else:
            print(f"[Epoch {epoch+1:03d}] Train Loss: {train_loss:.6f}")
        
        if (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)

    return model


# ------------------------------ #
#         Main Entrypoint        #
# ------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True, help="Subdirectory under demos/hdf5")
    parser.add_argument("--sequence_length", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=16)
    args = parser.parse_args()

    # --- Paths ---
    dataset_dir = "/home/vincenzopomponi/Documents/repos/DynaMimicGen/dmg/demos/hdf5"
    dir_path = os.path.join(dataset_dir, args.directory)
    dataset_path = os.path.join(dir_path, "dmp/low_dim.hdf5")

    seed = 0
    fix_random_seed(seed=seed)

    # --- Keys and Extraction ---
    with h5py.File(dataset_path, 'r') as f:
        demos = list(f['data'].keys())
        for demo in demos:
            demo_data = f['data'][demo]

            # Replicate the first action
            print(f"\n{demo}")
            print(f"actions: {demo_data['actions'].shape}")
            input(f"states: {demo_data['states'].shape}")

    obs_keys = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    data = extract_observations_from_robomimic(dataset_path, obs_keys, args.sequence_length)

    # --- Construct State Vector ---
    states = np.concatenate([
        data["object"],
        data["robot0_eef_pos"],
        data["robot0_eef_quat"],
        data["robot0_gripper_qpos"]
    ], axis=1)

    full_data = {
        "states": states,
        "actions": data["actions"]
    }

    # --- Windowing ---
    current_states, goal_targets = create_goal_prediction_windows(
        full_data, sequence_length=args.sequence_length, horizon=args.horizon
    )

    # print(f"Prepared training data:")
    # print(f"  States: {current_states.shape}")
    # print(f"  Goals:  {goal_targets.shape}")

    # --- Training ---

    lr = 1e-4

    wandb_options = dict(
    wandb_mode='online',  # "online", "offline" or "disabled"
    wandb_entity='vincenzo-pomponi-supsi',
    wandb_project=f"lgp"
    )

    wandb.init(
        project=wandb_options["wandb_project"],
        entity=wandb_options["wandb_entity"],
    )

    base_dir = "lgp_models/"
    save_dir = os.path.join(base_dir, args.directory)
    save_dir = os.path.join(save_dir, datetime.today().isoformat())
    
    os.makedirs(save_dir, exist_ok=True)
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_latent_goal_predictor(
        states=current_states,
        target_goals=goal_targets,
        state_dim=current_states.shape[1],
        hidden_dim=256,
        lr=lr,
        num_epochs=500,
        val_every=20,
        save_dir=save_dir,
        ckpt_dir=ckpt_dir,
        save_every=50,
    )

